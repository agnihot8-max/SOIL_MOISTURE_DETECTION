import os
import json
import smtplib
import requests
from email.message import EmailMessage
from datetime import datetime, timedelta
from dotenv import load_dotenv
from colorama import Fore, Style, init

init()
load_dotenv()

class AlertSystem:

    @staticmethod
    def check_configuration():
        ntfy_topic = os.environ.get('NTFY_TOPIC')
        if not ntfy_topic:
            print(f"  {Fore.YELLOW}[WARNING]{Style.RESET_ALL} NTFY_TOPIC is missing. Push notifications are disabled.")
        else:
            print(f"  {Fore.GREEN}[CONFIG]{Style.RESET_ALL} NTFY push notifications enabled for topic: {ntfy_topic}")

        smtp_email = os.environ.get('SMTP_EMAIL')
        smtp_password = os.environ.get('SMTP_PASSWORD')
        to_email = os.environ.get('MY_EMAIL')
        
        if not all([smtp_email, smtp_password, to_email]):
            print(f"  {Fore.YELLOW}[WARNING]{Style.RESET_ALL} SMTP credentials (SMTP_EMAIL, SMTP_PASSWORD, MY_EMAIL) are missing or incomplete. Email alerts are disabled.")
        else:
            print(f"  {Fore.GREEN}[CONFIG]{Style.RESET_ALL} Email alerts enabled. Sending to: {to_email}")

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
    def print_insight(node_name, timestamp, moisture, temp, batt, rssi, insight_data):
        print(f"{Fore.LIGHTBLACK_EX}[{timestamp}] {node_name} | Normal | Moist: {moisture:.0f} | Temp: {temp:.1f}C | Batt: {batt:.2f}v | RSSI: {rssi:.0f}dBm{Style.RESET_ALL}")
        print(f"   {Fore.CYAN}-> [AI Insight]{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}Status: NORMAL{Style.RESET_ALL}")
        z_val = insight_data['z_score']
        print(f"   {Fore.MAGENTA}-> [Z-Score]{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}{z_val:.2f} | The Z-Score measures sudden historical spikes; it must exceed 2.5 to trigger a statistical anomaly alert.{Style.RESET_ALL}")
        
        # ML Anomaly Score
        score = insight_data.get('if_score', 0.0)
        try:
            score_val = score.item() if hasattr(score, 'item') else float(score)
            print(f"   {Fore.GREEN}-> [ML Safety Score]{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}{score_val:.3f} | If this score drops below 0.0, the AI considers the multivariable data pattern highly suspicious.{Style.RESET_ALL}")
        except:
            pass

        std_dev = insight_data['std_dev']
        threshold = std_dev if std_dev > 50 else 50.0
        m_diff = abs(insight_data['m_diff'])
        t_diff = abs(insight_data['t_diff'])
        
        print(f"   {Fore.BLUE}-> [Moisture Deviation]{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}{m_diff:.2f} units | To prevent false alarms from minor soil variations, the AI requires this difference from the farm average to exceed {threshold:.2f} units before confirming an anomaly.{Style.RESET_ALL}")
        print(f"   {Fore.RED}-> [Temperature Deviation]{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}{t_diff:.2f}C | The AI requires a difference from the farm average of at least 2.00C before confirming a temperature anomaly.{Style.RESET_ALL}")
        
        # Expected Bounds
        exp_m = insight_data.get('expected_moist', moisture)
        std_m = insight_data.get('expected_moist_std', 0.0)
        lower_m = exp_m - (2.5 * std_m)
        upper_m = exp_m + (2.5 * std_m)
        
        exp_t = insight_data.get('expected_temp', temp)
        std_t = insight_data.get('expected_temp_std', 0.0)
        lower_t = exp_t - (2.5 * std_t)
        upper_t = exp_t + (2.5 * std_t)
        
        print(f"   {Fore.LIGHTCYAN_EX}-> [Expected Moisture]{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}{lower_m:.0f} to {upper_m:.0f} | Based on recent history, values outside this range will trigger a statistical anomaly.{Style.RESET_ALL}")
        print(f"   {Fore.LIGHTRED_EX}-> [Expected Temp]{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}{lower_t:.1f}C to {upper_t:.1f}C | Based on recent history, values outside this range will trigger a statistical anomaly.{Style.RESET_ALL}")
        
        # Absolute Farm Averages
        print(f"   {Fore.YELLOW}-> [Context]{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}Farm Averages | Moist: {insight_data['farm_avg_moist']:.0f} | Temp: {insight_data['farm_avg_temp']:.1f}C | Batt: {insight_data['farm_avg_batt']:.2f}v{Style.RESET_ALL}")
        
        # Seasonal Drift Indicator
        moist_drift = insight_data['drift_moist']
        temp_drift = insight_data['drift_temp']
        moist_str = f"{abs(moist_drift):.0f} units {'wetter' if moist_drift > 0 else 'drier'}"
        temp_str = f"{abs(temp_drift):.1f}C {'warmer' if temp_drift > 0 else 'colder'}"
        print(f"   {Fore.YELLOW}-> [Context]{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}Seasonal Drift | The soil is currently {temp_str} and {moist_str} than when the AI was trained.{Style.RESET_ALL}")

        # Live Weather Context
        has_rained = insight_data['has_rained']
        if has_rained != "Unknown":
            weather_str = "Recent Rain Confirmed" if has_rained == "True" else "No Recent Rain"
            print(f"   {Fore.YELLOW}-> [Context]{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}Weather API | {weather_str}{Style.RESET_ALL}")
            
        print("")

    @staticmethod
    def dispatch_external_alert(node_name, timestamp, reason, value):
        message_body = f" SOIL SENSOR ALERT \nNode: {node_name}\nAnomaly: {reason}\nTime: {timestamp}\nValue: {value}"
        
        AlertSystem.send_ntfy(message_body)
        AlertSystem.send_email(f"Alert: {node_name} - {reason}", message_body)

    @staticmethod
    def send_ntfy(body):
        ntfy_topic = os.environ.get('NTFY_TOPIC')
        
        if not ntfy_topic:
            print(f"  {Fore.YELLOW}[NTFY SKIPPED]{Style.RESET_ALL} No NTFY_TOPIC configured.")
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
            print(f"  {Fore.YELLOW}[EMAIL SKIPPED]{Style.RESET_ALL} Missing SMTP credentials.")
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

    @staticmethod
    def dispatch_eod_report(report_body):
        print(f"\n{Fore.GREEN}=== END OF DAY REPORT ==={Style.RESET_ALL}\n{report_body}")
        
        # Send NTFY
        ntfy_topic = os.environ.get('NTFY_TOPIC')
        if ntfy_topic:
            try:
                url = f"https://ntfy.sh/{ntfy_topic}"
                requests.post(url, data=report_body.encode(encoding='utf-8'), headers={
                    "Title": "Soil Moisture EOD Report",
                    "Priority": "default",
                    "Tags": "clipboard"
                })
            except Exception as e:
                print(f"  {Fore.RED}[NTFY FAILED]{Style.RESET_ALL} {e}")
                
        # Send Email
        AlertSystem.send_email("Soil Moisture EOD Report", report_body)
