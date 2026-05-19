import os
import json
import smtplib
import requests
from email.message import EmailMessage
from datetime import datetime, timedelta
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama for Windows support
init()
load_dotenv()

class AlertSystem:
    _last_alert_time = {}
    COOLDOWN_HOURS = 12
    STATE_FILE = "alert_state.json"
    _state_loaded = False

    @staticmethod
    def load_state():
        if AlertSystem._state_loaded:
            return
        if os.path.exists(AlertSystem.STATE_FILE):
            try:
                with open(AlertSystem.STATE_FILE, 'r') as f:
                    data = json.load(f)
                    AlertSystem._last_alert_time = {k: datetime.fromisoformat(v) for k, v in data.items()}
            except Exception as e:
                print(f"  {Fore.YELLOW}[WARNING]{Style.RESET_ALL} Failed to load alert state: {e}")
        AlertSystem._state_loaded = True

    @staticmethod
    def save_state():
        try:
            data = {k: v.isoformat() for k, v in AlertSystem._last_alert_time.items()}
            with open(AlertSystem.STATE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  {Fore.RED}[ERROR]{Style.RESET_ALL} Failed to save alert state: {e}")

    @staticmethod
    def log_status(node_name, entry_count, anomaly_count):
        print(f"\n{Style.BRIGHT}{Fore.CYAN}--- Node Report: {node_name} ---{Style.RESET_ALL}")
        print(f"Total Entries: {entry_count}")
        
        if anomaly_count == 0:
            print(f"Status: {Fore.GREEN}OK (No anomalies detected){Style.RESET_ALL}")
        else:
            print(f"Status: {Fore.YELLOW}WARNING ({anomaly_count} potential anomalies found){Style.RESET_ALL}")

    @staticmethod
    def print_anomaly(node_name, timestamp, reason, value):
        if value == "N/A" or value == "":
            print(f"  {Fore.RED}[ANOMALY]{Style.RESET_ALL} [{node_name}] {timestamp} | {reason}")
        else:
            print(f"  {Fore.RED}[ANOMALY]{Style.RESET_ALL} [{node_name}] {timestamp} | {reason} | Value: {value}")
            
        AlertSystem.dispatch_external_alert(node_name, timestamp, reason, value)
        
    @staticmethod
    def print_insight(node_name, timestamp, moisture, temp, batt, rssi, z_score, moisture_diff, temp_diff, std_dev):
        print(f"{Fore.LIGHTBLACK_EX}[{timestamp}] {node_name} | Normal | Moist: {moisture:.0f} | Temp: {temp:.1f}C | Batt: {batt:.2f}v | RSSI: {rssi:.0f}dBm{Style.RESET_ALL}")
        print(f"   {Fore.LIGHTBLACK_EX}-> [AI Insight] Status: NORMAL{Style.RESET_ALL}")
        print(f"   {Fore.LIGHTBLACK_EX}-> [Math] Z-Score is {z_score:.2f} (Must be > 2.5 to trigger alarm){Style.RESET_ALL}")
        
        threshold = std_dev if std_dev > 50 else 50.0
        print(f"   {Fore.LIGHTBLACK_EX}-> [Math] Farm Moisture Dev is {abs(moisture_diff):.2f} (If ML triggers, must be > {threshold:.2f} to pass minimum threshold){Style.RESET_ALL}")
        print(f"   {Fore.LIGHTBLACK_EX}-> [Math] Farm Temp Dev is {abs(temp_diff):.2f}C (If ML triggers, must be > 2.00C to pass minimum threshold){Style.RESET_ALL}")
        print("")

    @staticmethod
    def dispatch_external_alert(node_name, timestamp, reason, value):
        AlertSystem.load_state()
        alert_key = f"{node_name}_{reason}"
        now = datetime.now()
        
        # Rate limiting logic (12-hour cooldown per specific anomaly on a node)
        if alert_key in AlertSystem._last_alert_time:
            time_since_last = now - AlertSystem._last_alert_time[alert_key]
            if time_since_last < timedelta(hours=AlertSystem.COOLDOWN_HOURS):
                return # Still in cooldown period, don't spam
                
        AlertSystem._last_alert_time[alert_key] = now
        AlertSystem.save_state()
        
        message_body = f" SOIL SENSOR ALERT \nNode: {node_name}\nAnomaly: {reason}\nTime: {timestamp}\nValue: {value}"
        
        AlertSystem.send_ntfy(message_body)
        AlertSystem.send_email(f"Alert: {node_name} - {reason}", message_body)

    @staticmethod
    def send_ntfy(body):
        ntfy_topic = os.environ.get('NTFY_TOPIC')
        
        if not ntfy_topic:
            return
            
        try:
            url = f"https://ntfy.sh/{ntfy_topic}"
            requests.post(url, data=body.encode(encoding='utf-8'), headers={
                "Title": "Soil Moisture Anomaly",
                "Priority": "high"
            })
            print(f"  {Fore.GREEN}[NTFY]{Style.RESET_ALL} Push notification dispatched to topic: {ntfy_topic}")
        except Exception as e:
            print(f"  {Fore.RED}[NTFY FAILED]{Style.RESET_ALL} {e}")

    @staticmethod
    def send_email(subject, body):
        smtp_email = os.environ.get('SMTP_EMAIL')
        smtp_password = os.environ.get('SMTP_PASSWORD')
        to_email = os.environ.get('MY_EMAIL')
        
        if not all([smtp_email, smtp_password, to_email]):
            return
            
        try:
            msg = EmailMessage()
            msg.set_content(body)
            msg['Subject'] = subject
            msg['From'] = smtp_email
            msg['To'] = to_email

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(smtp_email, smtp_password)
                smtp.send_message(msg)
            print(f"  {Fore.GREEN}[EMAIL]{Style.RESET_ALL} Email alert dispatched to {to_email}")
        except Exception as e:
            print(f"  {Fore.RED}[EMAIL FAILED]{Style.RESET_ALL} Check credentials.")

    @staticmethod
    def log_info(message):
        print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {message}")

    @staticmethod
    def log_error(message):
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}")
