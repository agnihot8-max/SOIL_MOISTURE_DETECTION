# GEMINI.md

## 1. Project Goal / Scope

This project is an automated anomaly detection system for a soil moisture monitoring network.

The system should monitor recent sensor data, detect abnormal behavior, and eventually send alerts when something appears wrong.

The first version should prioritize a simple working prototype over a complex final system.

### Main goals

- Read recent sensor data from ThingSpeak
- Clean and organize the data
- Detect obvious sensor or node problems
- Use rolling averages and z-score checks for early anomaly detection
- Avoid false alerts during real environmental events, such as rainfall
- Keep the system modular so alerts, AI models, and dashboards can be added later

### Current scope

Include:
- Python-based data analysis
- ThingSpeak data fetching
- Basic rule-based anomaly detection
- Rolling-window statistics
- Console-based alerts first

Exclude for now:
- Full dashboard
- Complex AI models
- Permanent production deployment
- Hard-coded API keys
- Overly complicated architecture

### Development priority

Build in this order:

1. Get data from ThingSpeak
2. Convert it into a clean pandas DataFrame
3. Add basic anomaly checks
4. Add rolling average and z-score logic
5. Print clear alert messages
6. Add SMS/email alerts later
7. Add machine learning only after the basic system works