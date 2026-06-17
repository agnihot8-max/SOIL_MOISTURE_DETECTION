import os
import joblib
import json
from sklearn.ensemble import IsolationForest
from src.data_processor import DataProcessor
from src.alerts import AlertSystem

import pandas as pd
import numpy as np

# The base features and the new relative features
BASE_FEATURES = ['soil_moisture_raw', 'soil_temp_c', 'battery_voltage', 'rssi_dbm']
TRAIN_FEATURES = BASE_FEATURES + ['moisture_diff', 'temp_diff']

def train_isolation_forest(csv_paths):
    """
    Trains node-specific Isolation Forest models for decentralized anomaly detection.
    
    This function processes raw node data, aligns timestamps to build a unified
    network context, calculates farm-wide averages, and then trains and saves
    individual machine learning models and historical baselines for each node.
    """
    all_dfs = []
    
    for path in csv_paths:
        AlertSystem.log_info(f"Loading training data from {path}...")
        if not os.path.exists(path):
            AlertSystem.log_error(f"Training file not found: {path}")
            continue
            
        df = DataProcessor.load_csv(path)
        
        filename = os.path.basename(path).lower()
        if 'node_1' in filename: node_name = "Node 1"
        elif 'node_2' in filename: node_name = "Node 2"
        elif 'node_3' in filename: node_name = "Node 3"
        else: node_name = "Unknown"
        
        df['node_name'] = node_name
        all_dfs.append(df)
        
    if not all_dfs:
        AlertSystem.log_error("No training data could be loaded!")
        return

    aligned_dfs = []
    for i, df in enumerate(all_dfs):
        temp_df = df.set_index('created_at')[BASE_FEATURES].copy()
        temp_df.columns = [f"{c}_{i}" for c in BASE_FEATURES]
        aligned_dfs.append(temp_df)
        
    AlertSystem.log_info("Aligning timelines and calculating farm averages...")
    wide_df = pd.concat(aligned_dfs, axis=1).sort_index()
    wide_df = wide_df.ffill().bfill()
    
    moisture_cols = [f"soil_moisture_raw_{i}" for i in range(len(all_dfs))]
    temp_cols = [f"soil_temp_c_{i}" for i in range(len(all_dfs))]
    
    wide_df['farm_avg_moisture'] = wide_df[moisture_cols].mean(axis=1)
    wide_df['farm_avg_temp'] = wide_df[temp_cols].mean(axis=1)
    
    processed_dfs = []
    for i, df in enumerate(all_dfs):
        proc_df = df.set_index('created_at').copy()
        proc_df = proc_df.dropna(subset=BASE_FEATURES)
        
        proc_df['moisture_diff'] = proc_df['soil_moisture_raw'] - wide_df.loc[proc_df.index, 'farm_avg_moisture']
        proc_df['temp_diff'] = proc_df['soil_temp_c'] - wide_df.loc[proc_df.index, 'farm_avg_temp']
        
        processed_dfs.append(proc_df)

    train_df = pd.concat(processed_dfs).reset_index()
    
    for feature in TRAIN_FEATURES:
        if feature not in train_df.columns:
            AlertSystem.log_error(f"Missing required feature: {feature}")
            return
            
    X = train_df[TRAIN_FEATURES]

    os.makedirs('models', exist_ok=True)
    
    nodes = train_df['node_name'].unique()
    for node_name in nodes:
        node_df = train_df[train_df['node_name'] == node_name]
        X = node_df[TRAIN_FEATURES]

        AlertSystem.log_info(f"Training Decentralized Brain for {node_name} on {len(X)} entries...")
        
        model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        model.fit(X)
        
        safe_name = node_name.replace(" ", "_")
        model_path = os.path.join('models', f'{safe_name}_model.joblib')
        joblib.dump(model, model_path)
        
        node_means = X.mean().to_dict()
        means_path = os.path.join('models', f'{safe_name}_means.json')
        with open(means_path, 'w') as f:
            json.dump(node_means, f, indent=4)
            
        AlertSystem.log_info(f"Brain and Baselines saved for {node_name} to {model_path}")
        
    AlertSystem.log_status("Decentralized Training", len(train_df), 0)

if __name__ == "__main__":
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    training_files = [
        os.path.join(downloads_path, "node_1_8000E.csv"),
        os.path.join(downloads_path, "node_2_8000E.csv"),
        os.path.join(downloads_path, "node_3_8000E.csv")
    ]
    train_isolation_forest(training_files)
