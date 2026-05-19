import os
import joblib
from sklearn.ensemble import IsolationForest
from src.data_processor import DataProcessor
from src.alerts import AlertSystem

import pandas as pd
import numpy as np

# The base features and the new relative features
BASE_FEATURES = ['soil_moisture_raw', 'soil_temp_c', 'battery_voltage', 'rssi_dbm']
TRAIN_FEATURES = BASE_FEATURES + ['moisture_diff', 'temp_diff']

def train_isolation_forest(csv_paths):
    all_dfs = []
    
    for path in csv_paths:
        AlertSystem.log_info(f"Loading training data from {path}...")
        if not os.path.exists(path):
            AlertSystem.log_error(f"Training file not found: {path}")
            continue
        
        # Load and clean the dataset
        df = DataProcessor.load_csv(path)
        
        # Extract node name from filename
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

    # 1. Prepare dataframes for time alignment
    aligned_dfs = []
    for i, df in enumerate(all_dfs):
        # Ensure created_at is the index for time-based joining
        temp_df = df.set_index('created_at')[BASE_FEATURES].copy()
        # Rename columns so they don't collide when joined
        temp_df.columns = [f"{c}_{i}" for c in BASE_FEATURES]
        aligned_dfs.append(temp_df)
        
    # 2. Join all nodes on time to create a "Wide" farm snapshot
    AlertSystem.log_info("Aligning timelines and calculating farm averages...")
    wide_df = pd.concat(aligned_dfs, axis=1).sort_index()
    # Carry forward last known values to handle slight polling delays between nodes
    wide_df = wide_df.ffill().bfill()
    
    # 3. Calculate the Farm Average at every point in time
    moisture_cols = [f"soil_moisture_raw_{i}" for i in range(len(all_dfs))]
    temp_cols = [f"soil_temp_c_{i}" for i in range(len(all_dfs))]
    
    wide_df['farm_avg_moisture'] = wide_df[moisture_cols].mean(axis=1)
    wide_df['farm_avg_temp'] = wide_df[temp_cols].mean(axis=1)
    
    # 4. Reconstruct the training set with the relative features
    processed_dfs = []
    for i, df in enumerate(all_dfs):
        proc_df = df.set_index('created_at').copy()
        # Drop rows with missing base features first
        proc_df = proc_df.dropna(subset=BASE_FEATURES)
        
        # Calculate relative difference from the farm average at that exact time
        proc_df['moisture_diff'] = proc_df['soil_moisture_raw'] - wide_df.loc[proc_df.index, 'farm_avg_moisture']
        proc_df['temp_diff'] = proc_df['soil_temp_c'] - wide_df.loc[proc_df.index, 'farm_avg_temp']
        
        processed_dfs.append(proc_df)

    # Combine everything back into a massive 24,000-entry dataset
    train_df = pd.concat(processed_dfs).reset_index()
    
    # Ensure all required features exist and extract X
    for feature in TRAIN_FEATURES:
        if feature not in train_df.columns:
            AlertSystem.log_error(f"Missing required feature: {feature}")
            return
            
    X = train_df[TRAIN_FEATURES]

    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    import json
    
    # DECENTRALIZED TRAINING: Train a separate brain for each node
    nodes = train_df['node_name'].unique()
    for node_name in nodes:
        node_df = train_df[train_df['node_name'] == node_name]
        X = node_df[TRAIN_FEATURES]

        AlertSystem.log_info(f"Training Decentralized Brain for {node_name} on {len(X)} entries...")
        
        # Initialize Isolation Forest
        model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        model.fit(X)
        
        # Save the model
        safe_name = node_name.replace(" ", "_")
        model_path = os.path.join('models', f'{safe_name}_model.joblib')
        joblib.dump(model, model_path)
        
        # Calculate and save individual node means
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
