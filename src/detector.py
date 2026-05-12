import pandas as pd
import numpy as np
import joblib
import os
from src.alerts import AlertSystem

class AnomalyDetector:
    def __init__(self, window_size=5, z_threshold=3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        
        self.if_model = None
        model_path = os.path.join('models', 'master_if_model.joblib')
        if os.path.exists(model_path):
            self.if_model = joblib.load(model_path)
        else:
            AlertSystem.log_info("Isolation Forest model not found. Running in basic statistical mode.")

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

    def detect_isolation_forest(self, df, network_context=None):
        """Uses the pre-trained Isolation Forest to find multivariate anomalies and contextualizes them."""
        results = pd.DataFrame(index=df.index)
        results['if_anomaly'] = False
        results['if_reason'] = ""
        
        if self.if_model is None:
            return results
            
        base_features = ['soil_moisture_raw', 'soil_temp_c', 'battery_voltage', 'rssi_dbm']
        train_features = base_features + ['moisture_diff', 'temp_diff']
        
        # Ensure we have the required base features
        if not all(feature in df.columns for feature in base_features):
            return results
            
        # Calculate dynamic farm average using network context
        df = df.copy()
        df['moisture_diff'] = 0.0
        df['temp_diff'] = 0.0
        
        if network_context:
            for idx in df.index:
                target_time = df.loc[idx, 'created_at']
                moisture_vals = [df.loc[idx, 'soil_moisture_raw']]
                temp_vals = [df.loc[idx, 'soil_temp_c']]
                
                # Get the closest reading from other nodes
                for name, other_df in network_context.items():
                    if other_df is not None and not other_df.empty:
                        time_diffs = (other_df['created_at'] - target_time).abs()
                        if time_diffs.min() < pd.Timedelta(hours=2):
                            closest_idx = time_diffs.idxmin()
                            moisture_vals.append(other_df.loc[closest_idx, 'soil_moisture_raw'])
                            temp_vals.append(other_df.loc[closest_idx, 'soil_temp_c'])
                
                # Calculate farm average at this exact point in time
                farm_avg_moist = sum(moisture_vals) / len(moisture_vals)
                farm_avg_temp = sum(temp_vals) / len(temp_vals)
                
                df.loc[idx, 'moisture_diff'] = df.loc[idx, 'soil_moisture_raw'] - farm_avg_moist
                df.loc[idx, 'temp_diff'] = df.loc[idx, 'soil_temp_c'] - farm_avg_temp
            
        # Predict (-1 is anomaly, 1 is normal)
        predictions = self.if_model.predict(df[train_features].fillna(0))
        
        # Calculate historical averages for context
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
                    reason = "Battery Voltage Unusually Low/Dead" if row['battery_voltage'] < means['battery_voltage'] else "Battery Voltage Unusually High"
                elif top_feature == 'soil_temp_c' or top_feature == 'temp_diff':
                    reason = "Temperature Unusually High" if row['soil_temp_c'] > means['soil_temp_c'] else "Temperature Unusually Low"
                elif top_feature == 'soil_moisture_raw' or top_feature == 'moisture_diff':
                    reason = "Moisture Deviation from Farm Average"
                else:
                    reason = f"Network Anomaly (RSSI: {row['rssi_dbm']})"
                    
                results.iloc[idx, results.columns.get_loc('if_reason')] = reason
                
        return results

    def analyze(self, df, network_context=None):
        """Runs all detection logic on the DataFrame, using network context if available."""
        results = pd.DataFrame(index=df.index)
        
        # 1. Statistical Anomalies
        results['zscore_anomaly'] = self.calculate_zscore(df['soil_moisture_raw'])
        
        # 2. Sensor Stuck
        results['stuck_sensor'] = self.detect_flat_line(df['soil_moisture_raw'])
        
        # 3. Range Checks
        range_checks = self.detect_out_of_range(df)
        results = pd.concat([results, range_checks], axis=1)
        
        # 4. Machine Learning Anomaly (Isolation Forest)
        if_results = self.detect_isolation_forest(df, network_context)
        results = pd.concat([results, if_results], axis=1)
        
        # 5. Rainfall Heuristic & Cross-Node Validation
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
                            # Check if the other node also saw a significant moisture increase nearby
                            # We check a small window around the closest index
                            window_start = max(0, other_df.index.get_loc(closest_idx) - 2)
                            window_end = min(len(other_df), other_df.index.get_loc(closest_idx) + 3)
                            
                            other_diffs = other_df['soil_moisture_raw'].diff().iloc[window_start:window_end]
                            if (other_diffs > (other_df['soil_moisture_raw'].mean() * 0.05)).any():
                                spiked_count += 1
                                
                    # If at least one other node spiked, it's a global event (Rain)
                    if spiked_count >= 1:
                        results.loc[idx, 'is_global_event'] = True
                        
                        # Suppress ML anomaly if it was just complaining about the moisture spike
                        if results.loc[idx, 'if_reason'] == "Moisture Deviation from Farm Average":
                            results.loc[idx, 'if_anomaly'] = False
                            results.loc[idx, 'if_reason'] = ""
                            
            # Anomaly is true if Z-score is high AND it's NOT a global event
            results['final_anomaly'] = results['zscore_anomaly'] & ~results['is_global_event']
        else:
            # Fallback to basic single-node logic
            results['final_anomaly'] = results['zscore_anomaly'] & ~results['is_spike']
        
        return results
