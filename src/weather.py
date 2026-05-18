import requests
from src.alerts import AlertSystem

class WeatherContext:
    @staticmethod
    def check_recent_precipitation(lat, lon):
        """
        Checks the Open-Meteo API for precipitation at the given coordinates over the last 48 hours.
        Returns True if rain > 0, False otherwise. Returns None if API fails.
        """
        if not lat or not lon:
            return None
            
        try:
            # We request hourly precipitation for the past 6 hours for a much tighter verification window
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation&timezone=auto&past_hours=6&forecast_hours=1"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            precip_hourly = data.get('hourly', {}).get('precipitation', [])
            
            # Filter out None values just in case
            valid_precip = [p for p in precip_hourly if p is not None]
            
            if not valid_precip:
                return None
                
            total_rain_mm = sum(valid_precip)
            
            AlertSystem.log_info(f"Weather API: {total_rain_mm:.2f}mm of rain detected over the last 6 hours.")
            
            return total_rain_mm > 0.0
            
        except Exception as e:
            AlertSystem.log_error(f"Weather API failed: {e}")
            return None
