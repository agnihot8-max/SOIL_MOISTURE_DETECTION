import pandas as pd
import numpy as np
import joblib
import os
from src.alerts import AlertSystem
from src.weather import WeatherContext

class AnomalyDetector:
    def __init__(self, window_size=5, z_threshold=3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        
        self.models = {}
        self.means = {}

    def detect_flat_line(self, series, window=5):
        """Detects if the sensor values are stuck (no change)."""
        return series.rolling(window=window).std() == 0

    def detect_out_of_range(self, df):
        """Flags values that are physically unrealistic."""
        anomalies = pd.DataFrame(index=df.index)
        anomalies['moisture_range'] = (df['soil_moisture_raw'] < 0) | (df['soil_moisture_raw'] > 4000) # Typical raw range
        anomalies['battery_low'] = df['battery_voltage'] < 3.0
        return anomalies

    def calculate_zscore(self, series):
        """Calculates rolling Z-score."""
        rolling_mean = series.rolling(window=self.window_size).mean()
        rolling_std = series.rolling(window=self.window_size).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        z_scores = (series - rolling_mean) / rolling_std
        return z_scores.abs() > self.z_threshold

    def detect_isolation_forest(self, df, node_name, network_context=None):
        """Uses the pre-trained Isolation Forest to find multivariate anomalies and contextualizes them."""
        results = pd.DataFrame(index=df.index)
        results['if_anomaly'] = False
        results['if_reason'] = ""
        
        safe_name = node_name.replace(" ", "_")
        model_path = os.path.join('models', f'{safe_name}_model.joblib')
        means_path = os.path.join('models', f'{safe_name}_means.json')
        
        if safe_name not in self.models:
            if os.path.exists(model_path) and os.path.exists(means_path):
                self.models[safe_name] = joblib.load(model_path)
                import json
                with open(means_path, 'r') as f:
                    self.means[safe_name] = json.load(f)
            else:
                return results
                
        node_model = self.models[safe_name]
        node_means = self.means[safe_name]
            
        base_features = ['soil_moisture_raw', 'soil_temp_c', 'battery_voltage', 'rssi_dbm']
        train_features = base_features + ['moisture_diff', 'temp_diff']
        
        if not all(feature in df.columns for feature in base_features):
            return results
            
        df = df.copy()
        df['moisture_diff'] = 0.0
        df['temp_diff'] = 0.0
        df['battery_diff'] = 0.0
        
        if network_context:
            for idx in df.index:
                target_time = df.loc[idx, 'created_at']
                moisture_vals = [df.loc[idx, 'soil_moisture_raw']]
                temp_vals = [df.loc[idx, 'soil_temp_c']]
                battery_vals = [df.loc[idx, 'battery_voltage']]
                
                # Get the closest reading from other nodes
                for name, other_df in network_context.items():
                    if other_df is not None and not other_df.empty:
                        time_diffs = (other_df['created_at'] - target_time).abs()
                        if time_diffs.min() < pd.Timedelta(hours=2):
                            closest_idx = time_diffs.idxmin()
                            moisture_vals.append(other_df.loc[closest_idx, 'soil_moisture_raw'])
                            temp_vals.append(other_df.loc[closest_idx, 'soil_temp_c'])
                            battery_vals.append(other_df.loc[closest_idx, 'battery_voltage'])
                
                farm_avg_moist = sum(moisture_vals) / len(moisture_vals)
                farm_avg_temp = sum(temp_vals) / len(temp_vals)
                farm_avg_batt = sum(battery_vals) / len(battery_vals)
                
                df.loc[idx, 'moisture_diff'] = df.loc[idx, 'soil_moisture_raw'] - farm_avg_moist
                df.loc[idx, 'temp_diff'] = df.loc[idx, 'soil_temp_c'] - farm_avg_temp
                df.loc[idx, 'battery_diff'] = df.loc[idx, 'battery_voltage'] - farm_avg_batt
            
        predictions = node_model.predict(df[train_features].fillna(0))
        
        # Use the live rolling mean for context extraction, NOT historical training means!
        # Historical means will cause mathematically invalid Z-scores if the season has shifted.
        means = df[train_features].mean()
        stds = df[train_features].std().replace(0, 1e-9) # avoid div zero
        
        for idx in range(len(predictions)):
            if predictions[idx] == -1:
                results.iloc[idx, results.columns.get_loc('if_anomaly')] = True
                
                # Contextualize: Which feature is furthest from the mean?
                row = df.iloc[idx][train_features]
                z_scores = ((row - means) / stds).abs()
                
                top_feature = z_scores.idxmax()
                
                # Assign a smart reason based on the top feature
                if top_feature == 'battery_voltage':
                    val = row['battery_voltage']
                    exp = means['battery_voltage']
                    
                    battery_diff = df.loc[df.index[idx], 'battery_diff'] if 'battery_diff' in df.columns else 0.0
                    
                    if abs(val - exp) < 0.2 or (network_context and abs(battery_diff) < 0.2):
                        results.iloc[idx, results.columns.get_loc('if_anomaly')] = False
                        reason = ""
                    else:
                        reason = f"Battery Voltage Unusually Low (Act: {val:.2f}v, Exp: {exp:.2f}v)" if val < exp else f"Battery Voltage Unusually High (Act: {val:.2f}v, Exp: {exp:.2f}v)"
                elif top_feature == 'soil_temp_c' or top_feature == 'temp_diff':
                    if network_context and abs(row['temp_diff']) < 2.0:
                        results.iloc[idx, results.columns.get_loc('if_anomaly')] = False
                        reason = ""
                    else:
                        val = row['soil_temp_c']
                        exp = means['soil_temp_c']
                        reason = f"Temperature Unusually High (Act: {val:.1f}C, Exp: {exp:.1f}C)" if val > exp else f"Temperature Unusually Low (Act: {val:.1f}C, Exp: {exp:.1f}C)"
                elif top_feature == 'soil_moisture_raw' or top_feature == 'moisture_diff':
                    rolling_std = df['soil_moisture_raw'].std()
                    threshold = rolling_std if rolling_std > 50 else 50.0
                    
                    if abs(row['moisture_diff']) < threshold:
                        results.iloc[idx, results.columns.get_loc('if_anomaly')] = False
                        reason = ""
                    else:
                        val = row['soil_moisture_raw']
                        exp = means['soil_moisture_raw']
                        reason = f"Moisture Deviation from Farm Average (Act: {val:.0f}, Exp: {exp:.0f})"
                else:
                    reason = f"Network Anomaly (RSSI: {row['rssi_dbm']:.0f} dBm, Exp: {means['rssi_dbm']:.0f} dBm)"
                    
                results.iloc[idx, results.columns.get_loc('if_reason')] = reason
                
        # Attach raw math variables for Verbose Insight Engine
        results['raw_moisture_diff'] = df['moisture_diff']
        results['raw_temp_diff'] = df['temp_diff']
        results['raw_std_dev'] = df['soil_moisture_raw'].std()
                
        return results

    def analyze(self, df, node_name, network_context=None):
        """Runs all detection logic on the DataFrame, using network context if available."""
        results = pd.DataFrame(index=df.index)
        
        # 1. Statistical Anomalies
        z_scores = ((df['soil_moisture_raw'] - df['soil_moisture_raw'].rolling(window=self.window_size, min_periods=1).mean()) / 
                    df['soil_moisture_raw'].rolling(window=self.window_size, min_periods=1).std().replace(0, 1e-9)).abs()
        results['zscore_anomaly'] = z_scores > self.z_threshold
        results['raw_zscore'] = z_scores # Save for Insight Engine
        
        # 2. Sensor Stuck
        results['stuck_sensor'] = self.detect_flat_line(df['soil_moisture_raw'])
        
        # 3. Range Checks
        range_checks = self.detect_out_of_range(df)
        results = pd.concat([results, range_checks], axis=1)
        
        if_results = self.detect_isolation_forest(df, node_name, network_context)
        results = pd.concat([results, if_results], axis=1)
        
        lat = os.environ.get('FARM_LATITUDE')
        lon = os.environ.get('FARM_LONGITUDE')
        
        moisture_diff = df['soil_moisture_raw'].diff()
        results['is_spike'] = moisture_diff > (df['soil_moisture_raw'].mean() * 0.05)
        results['is_global_event'] = False
        
        if network_context:
            for idx in results.index:
                if results.loc[idx, 'is_spike']:
                    time_of_spike = df.loc[idx, 'created_at']
                    spiked_count = 0
                    
                    for node_name, other_df in network_context.items():
                        if other_df is None or other_df.empty:
                            continue
                            
                        # Find closest data point in time (within 2 hours)
                        time_diffs = (other_df['created_at'] - time_of_spike).abs()
                        if time_diffs.min() < pd.Timedelta(hours=2):
                            closest_idx = time_diffs.idxmin()
                            window_start = max(0, other_df.index.get_loc(closest_idx) - 2)
                            window_end = min(len(other_df), other_df.index.get_loc(closest_idx) + 3)
                            
                            other_diffs = other_df['soil_moisture_raw'].diff().iloc[window_start:window_end]
                            if (other_diffs > (other_df['soil_moisture_raw'].mean() * 0.05)).any():
                                spiked_count += 1
                                
                    # Cross-reference with the Weather API
                    has_rained = WeatherContext.check_recent_precipitation(lat, lon)
                    
                    if has_rained:
                        AlertSystem.log_info("Weather API confirms recent rainfall. Suppressing anomalies.")
                        results.loc[idx, 'is_global_event'] = True
                        if results.loc[idx, 'if_reason'] == "Moisture Deviation from Farm Average":
                            results.loc[idx, 'if_anomaly'] = False
                            results.loc[idx, 'if_reason'] = ""
                    elif has_rained is False:
                        AlertSystem.log_info("Moisture spiked but Weather API reports NO RAIN! Possible irrigation or leak.")
                        results.loc[idx, 'is_global_event'] = False
                    else:
                        # Weather API failed or coordinates missing. Fallback to node correlation.
                        if spiked_count >= 1:
                            AlertSystem.log_info("Weather API unavailable. Using cross-node validation: Global event detected.")
                            results.loc[idx, 'is_global_event'] = True
                            
                            if results.loc[idx, 'if_reason'] == "Moisture Deviation from Farm Average":
                                results.loc[idx, 'if_anomaly'] = False
                                results.loc[idx, 'if_reason'] = ""
                                
            # Anomaly is true if Z-score is high AND it's NOT a global event
            results['final_anomaly'] = results['zscore_anomaly'] & ~results['is_global_event']
        else:
            # Fallback to basic single-node logic
            results['final_anomaly'] = results['zscore_anomaly'] & ~results['is_spike']
        
        return results
