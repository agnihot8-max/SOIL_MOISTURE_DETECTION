import pandas as pd
import os

class DataProcessor:
    COLUMN_MAPPING = {
        'field1': 'bus_id',
        'field2': 'soil_moisture_raw',
        'field3': 'soil_temp_c',
        'field4': 'battery_voltage',
        'field5': 'rssi_dbm',
        'field6': 'snr_db'
    }

    @staticmethod
    def _clean_dataframe(df):
        # Rename columns based on the provided mapping
        df = df.rename(columns=DataProcessor.COLUMN_MAPPING)
        
        # Convert created_at to datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Ensure numeric types for sensors
        numeric_cols = ['soil_moisture_raw', 'soil_temp_c', 'battery_voltage', 'rssi_dbm', 'snr_db']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop entries with missing moisture (critical for anomaly detection)
        df = df.dropna(subset=['soil_moisture_raw'])
        
        # Sort by timestamp
        df = df.sort_values('created_at')
        
        return df

    @staticmethod
    def load_csv(file_path):
        """Loads and cleans a ThingSpeak export CSV."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)
        return DataProcessor._clean_dataframe(df)

    @staticmethod
    def load_json(json_data):
        """Loads and cleans a ThingSpeak API JSON response ('feeds' list)."""
        df = pd.DataFrame(json_data)
        return DataProcessor._clean_dataframe(df)

    @staticmethod
    def get_summary(df):
        """Returns basic statistics for the dataset."""
        return df.describe()
