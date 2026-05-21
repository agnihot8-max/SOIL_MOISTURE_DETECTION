import os
import time
import argparse
import subprocess
import requests
from src.data_processor import DataProcessor
from src.detector import AnomalyDetector
from src.alerts import AlertSystem

# Configuration for Live Mode
CHANNELS = {
    "Node 1": "2996904",
    "Node 2": "2996908",
    "Node 3": "2996910"
}
RESULTS_TO_FETCH = 20
POLL_INTERVAL_SECONDS = 60

def process_node_df(df, node_name, network_context=None):
    """Processes a loaded DataFrame with network context."""
    try:
        AlertSystem.log_info(f"Analyzing data for {node_name}...")
        
        detector = AnomalyDetector(window_size=10, z_threshold=2.5)
        analysis_results = detector.analyze(df, node_name, network_context)
        
        anomalies = analysis_results[analysis_results['final_anomaly'] == True]
        if_anomalies = analysis_results[analysis_results['if_anomaly'] == True]
        
        AlertSystem.log_status(node_name, len(df), len(anomalies) + len(if_anomalies))
        
        if not anomalies.empty:
            for idx, row in anomalies.iterrows():
                AlertSystem.print_anomaly(
                    node_name,
                    df.loc[idx, 'created_at'], 
                    "Statistical Outlier (Z-Score)", 
                    df.loc[idx, 'soil_moisture_raw']
                )
                
        if not if_anomalies.empty:
            for idx, row in if_anomalies.iterrows():
                AlertSystem.print_anomaly(
                    node_name,
                    df.loc[idx, 'created_at'], 
                    f"ML Anomaly: {row['if_reason']}", 
                    "N/A (Multivariate)"
                )
        
        stuck = analysis_results[analysis_results['stuck_sensor'] == True]
        if not stuck.empty:
            for idx, row in stuck.iterrows():
                AlertSystem.print_anomaly(node_name, df.loc[idx, 'created_at'], "Sensor Flat-line (Possible failure)", df.loc[idx, 'soil_moisture_raw'])
                
        battery_low = analysis_results[analysis_results['battery_low'] == True]
        if not battery_low.empty:
            for idx, row in battery_low.iterrows():
                AlertSystem.print_anomaly(node_name, df.loc[idx, 'created_at'], "Low Battery", df.loc[idx, 'battery_voltage'])

    except Exception as e:
        AlertSystem.log_error(f"Failed to process {node_name}: {str(e)}")

def run_batch_mode():
    """Runs the analysis on local CSV files with network context."""
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    files_to_process = {
        "Node A": os.path.join(downloads_path, "feed.csv"),
        "Node B": os.path.join(downloads_path, "feed (1).csv")
    }
    
    # 1. Load all available data to form the network context
    network_dfs = {}
    for node_name, file_path in files_to_process.items():
        if os.path.exists(file_path):
            try:
                network_dfs[node_name] = DataProcessor.load_csv(file_path)
            except Exception as e:
                AlertSystem.log_error(f"Failed to load {file_path}: {e}")
        else:
            AlertSystem.log_error(f"File not found for {node_name}: {file_path}")
            
    # 2. Analyze each node using the others as context
    for node_name, df in network_dfs.items():
        context = {name: other_df for name, other_df in network_dfs.items() if name != node_name}
        process_node_df(df, node_name, context)

def run_live_mode(single_run=False, detail=False):
    """Continuously polls ThingSpeak API for all channels and performs cross-node validation."""
    mode_text = "stateless cron job" if single_run else "live monitoring"
    AlertSystem.log_info(f"Starting {mode_text} for {len(CHANNELS)} nodes...")
    
    last_entry_ids = {node: None for node in CHANNELS}
    detector = AnomalyDetector(window_size=10, z_threshold=2.5)

    while True:
        try:
            # 1. Fetch data from all channels
            network_dfs = {}
            latest_entries = {}
            
            for node_name, channel_id in CHANNELS.items():
                url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?results={RESULTS_TO_FETCH}"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                
                feeds = data.get("feeds", [])
                if feeds:
                    network_dfs[node_name] = DataProcessor.load_json(feeds)
                    latest_entries[node_name] = feeds[-1]
                else:
                    AlertSystem.log_error(f"No feeds returned for {node_name}.")
                    
                import time
                time.sleep(1.5) # Respect ThingSpeak Free Tier rate limit (1 req/sec)
                    
            # 2. Analyze each node using the network context
            for node_name, df in network_dfs.items():
                current_entry_id = latest_entries[node_name].get("entry_id")
                
                if current_entry_id != last_entry_ids[node_name]:
                    context = {name: other_df for name, other_df in network_dfs.items() if name != node_name}
                    
                    if len(df) >= detector.window_size:
                        analysis_results = detector.analyze(df, node_name, context)
                        
                        latest_status = analysis_results.iloc[-1]
                        latest_time = df.iloc[-1]['created_at']
                        latest_value = df.iloc[-1]['soil_moisture_raw']
                        latest_temp = df.iloc[-1]['soil_temp_c']
                        latest_batt = df.iloc[-1]['battery_voltage']
                        latest_rssi = df.iloc[-1]['rssi_dbm']

                        is_anomaly = latest_status['if_anomaly'] or latest_status['final_anomaly'] or latest_status['stuck_sensor'] or latest_status['battery_low']
                        
                        if detail and not is_anomaly:
                            insight_data = {
                                'z_score': latest_status['raw_zscore'],
                                'm_diff': latest_status['raw_moisture_diff'],
                                't_diff': latest_status['raw_temp_diff'],
                                'std_dev': latest_status['raw_std_dev'],
                                'if_score': latest_status.get('if_score', 0.0),
                                'farm_avg_moist': latest_status.get('farm_avg_moist', latest_value),
                                'farm_avg_temp': latest_status.get('farm_avg_temp', latest_temp),
                                'farm_avg_batt': latest_status.get('farm_avg_batt', latest_batt),
                                'drift_moist': latest_status.get('drift_moist', 0.0),
                                'drift_temp': latest_status.get('drift_temp', 0.0),
                                'has_rained': latest_status.get('has_rained', "Unknown"),
                                'expected_moist': latest_status.get('expected_moist', 0.0),
                                'expected_moist_std': latest_status.get('expected_moist_std', 0.0),
                                'expected_temp': latest_status.get('expected_temp', 0.0),
                                'expected_temp_std': latest_status.get('expected_temp_std', 0.0)
                            }
                            AlertSystem.print_insight(node_name, latest_time, latest_value, latest_temp, latest_batt, latest_rssi, insight_data)
                        else:
                            print(f"[{latest_time}] {node_name} | New Entry (ID: {current_entry_id}) | Moist: {latest_value:.0f} | Temp: {latest_temp:.1f}C | Batt: {latest_batt:.2f}v | RSSI: {latest_rssi:.0f}dBm")

                        if latest_status['is_global_event']:
                            AlertSystem.log_info(f"Verified global event (Rain) for {node_name}. Anomalies suppressed.")

                        if latest_status['if_anomaly']:
                            AlertSystem.print_anomaly(node_name, latest_time, f"ML Anomaly: {latest_status['if_reason']}", "N/A")
                        elif latest_status['final_anomaly']:
                            AlertSystem.print_anomaly(node_name, latest_time, "Statistical Outlier (Z-Score)", latest_value)
                        
                        if latest_status['stuck_sensor']:
                            AlertSystem.print_anomaly(node_name, latest_time, "Sensor Flat-line (Possible failure)", latest_value)
                            
                        if latest_status['battery_low']:
                            AlertSystem.print_anomaly(node_name, latest_time, "Low Battery", df.iloc[-1]['battery_voltage'])
                    else:
                        AlertSystem.log_info(f"Building initial window buffer for {node_name}... ({len(df)}/{detector.window_size})")

                    last_entry_ids[node_name] = current_entry_id

            if single_run:
                AlertSystem.log_info("Cron execution completed. Exiting.")
                break

            time.sleep(POLL_INTERVAL_SECONDS)

        except requests.exceptions.RequestException as e:
            AlertSystem.log_error(f"Network error: {e}")
            if not single_run: time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            AlertSystem.log_error(f"Live loop error: {str(e)}")
            if not single_run: time.sleep(POLL_INTERVAL_SECONDS)
            else: break

def main():
    parser = argparse.ArgumentParser(description="Soil Moisture Anomaly Detection")
    parser.add_argument('--mode', choices=['batch', 'live', 'train', 'cron'], default='batch',
                        help="Run mode: 'batch' for local CSVs, 'live' for ThingSpeak API polling, 'train' to train the IF model, 'cron' for single run.")
    parser.add_argument('--detail', action='store_true', help="Enable the Insight Engine to print detailed math for normal readings.")
    
    args = parser.parse_args()
    
    # Check alert configuration before running any modes
    AlertSystem.check_configuration()
    
    if args.mode == 'train':
        subprocess.run(["python", "-m", "src.train_model"])
    elif args.mode == 'live':
        run_live_mode(single_run=False, detail=args.detail)
    elif args.mode == 'cron':
        run_live_mode(single_run=True, detail=args.detail)
    else:
        run_batch_mode()

if __name__ == "__main__":
    main()
