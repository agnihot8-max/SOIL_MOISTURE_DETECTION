import os
import pandas as pd
from src.data_processor import DataProcessor
from src.detector import AnomalyDetector
from src.alerts import AlertSystem

def test_anomaly():
    # Load a normal dataset
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    file_path = os.path.join(downloads_path, "feed.csv")
    
    if not os.path.exists(file_path):
        print(f"Could not find {file_path}")
        return
        
    df = DataProcessor.load_csv(file_path)
    
    print("\n--- INJECTING FAKE ANOMALY ---")
    print("Simulating a sudden irrigation pipe burst on Node A...")
    
    # Intentionally corrupt the very last reading to be a massive spike
    last_idx = df.index[-1]
    df.loc[last_idx, 'soil_moisture_raw'] = 3500  # Massive jump!
    
    # We need to simulate the rest of the farm staying dry (moisture around 500)
    # so the AI sees that Node A's massive jump is completely isolated.
    mock_node_b = df.copy()
    mock_node_b['soil_moisture_raw'] = 500
    
    mock_node_c = df.copy()
    mock_node_c['soil_moisture_raw'] = 500
    
    network_context = {
        "Node B": mock_node_b,
        "Node C": mock_node_c
    }
    
    print("Running AI Detector...\n")
    detector = AnomalyDetector(window_size=10, z_threshold=2.5)
    
    analysis_results = detector.analyze(df, network_context=network_context)
    
    latest_status = analysis_results.iloc[-1]
    
    if latest_status['if_anomaly']:
        AlertSystem.print_anomaly("Node A", df.iloc[-1]['created_at'], f"ML Anomaly: {latest_status['if_reason']}", df.iloc[-1]['soil_moisture_raw'])
    elif latest_status['final_anomaly']:
        AlertSystem.print_anomaly("Node A", df.iloc[-1]['created_at'], "Statistical Outlier", df.iloc[-1]['soil_moisture_raw'])
    else:
        print("No anomaly detected. (Something is wrong with the AI!)")

if __name__ == "__main__":
    test_anomaly()
