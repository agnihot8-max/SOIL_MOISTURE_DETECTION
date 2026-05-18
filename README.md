# Multi-Node Soil Moisture Anomaly Detection System

An automated, data-driven monitoring application built to track soil moisture levels, cross-validate readings across multiple farm nodes, and instantly dispatch intelligent alerts when anomalies or hardware failures occur.

The system pulls live sensor data from online **ThingSpeak API** channels or uses historical spreadsheet exports to run a dual-layered anomaly detection pipeline.

---

## 🚀 Key Features

* **Dual-Engine Anomaly Detection:**
  * **Statistical Outliers (Rolling Z-Score):** Tracks standard mathematical deviations to immediately catch sudden, sharp moisture spikes or drops on an individual sensor.
  * **Machine Learning Outliers (Isolation Forest):** Uses a pre-trained Scikit-learn AI model to scan raw measurements and cross-node differences (`moisture_diff`, `temp_diff`) together, catching complex, hidden environmental problems that basic boundary rules might miss.
* **Smart Cross-Node Validation (Rainfall Suppression):** Evaluates neighboring sensor nodes across the network. If a massive moisture spike occurs across multiple nodes within a tight time window, the system recognizes it as a global weather event (e.g., Rain) and intelligently suppresses false-positive alerts.
* **Hardware & Battery Diagnostics:** Automatically identifies frozen sensor telemetry (flat-lined readings indicating sensor failure) and captures critical low-voltage battery thresholds (< 3.0V).
* **Multi-Channel Dispatcher with Rate-Limiting:** Dispatches urgent alerts via phone push notifications (**ntfy.sh**) and **SMTP Email**, utilizing a built-in 12-hour per-anomaly cooldown period to prevent notification spam.
* **Automated CI/CD State Tracking:** Run as a continuous background script, a manual batch file processor, or an automated stateless **GitHub Actions cron job** (scheduled every 31 minutes) that dynamically saves active alert tracking into `alert_state.json`.

---

## 📁 Repository Structure

* **Automated CI/CD Pipelines:**
  * **GitHub Workflows:** The configuration file located in `.github/workflows/soil_monitor.yml` that automates the execution engine every 31 minutes via cron schedules.
* **Saved Model Files:**
  * **Master ML Model:** Saved AI model parameters located in `models/master_if_model.joblib` containing the pre-trained Isolation Forest binaries.
* **Core Application Codebase:**
  * **System Initialization:** Module anchor located in `src/__init__.py` making the internal source directory importable.
  * **Notification Engine:** Dispatch modules located in `src/alerts.py` handling multi-channel ntfy and SMTP email alerting alongside persistent state caching.
  * **Data Ingestion:** Cleaning modules located in `src/data_processor.py` handling ThingSpeak API JSON parsing, CSV loading, and column mapping.
  * **Detection Analytics:** Core algorithmic logic located in `src/detector.py` executing rolling Z-score passes, Isolation Forest AI predictions, and rainfall filtering rules.
  * **Training Infrastructure:** Script modules located in `src/train_model.py` that handle spatial-temporal timeline alignments across farm nodes to output updated model files.
* **Configuration & Execution Entrypoints:**
  * **Environment Blueprints:** Configuration templates located in `.env.example` defining required API endpoints, SMTP credentials, and alert topic strings.
  * **Git Exclusions:** Tracking configuration located in `.gitignore` protecting local environments, cached states, and virtual environments from tracking.
  * **Persistent Cache Engine:** Active tracking logs located in `alert_state.json` preventing alert spam by caching active cooldown timestamps.
  * **Application Main Interface:** Core execution engine located in `main.py` serving as the multi-mode command-line driver for live background monitoring, batch processing, cron tasks, or training pipelines.
  * **Third-Party Dependencies:** Pinpoint definitions located in `requirements.txt` listing mandatory packages like pandas, scikit-learn, joblib, and requests.
  * **Validation Suite:** Local testing frameworks located in `test_anomaly.py` running unit checks against the anomaly detection modules.

---

## 🛠️ Getting Started

* **Infrastructure Prerequisites:**
  * **Python Runtime:** Python 3.11 or higher installed on your target deployment environment.
  * **Push Gateway Topic:** A custom, secure topic string registered on the public [ntfy.sh](https://ntfy.sh) server gateway.
  * **SMTP Relay Service:** An active SMTP mail provider context (such as a Gmail App Password) to distribute system dispatches.
* **Repository Deployment:**
  * **Source Cloning:** Run `git clone https://github.com/agnihot8-max/SOIL_MOISTURE_DETECTION.git` to mirror the tracking assets locally.
  * **Directory Context:** Run `cd SOIL_MOISTURE_DETECTION` to step into the main root workspace.
* **Virtualization Setup:**
  * **Environment Allocation:** Run `python -m venv venv` to isolate system libraries from your local environment.
  * **Unix Core Activation:** Run `source venv/bin/activate` on Linux or macOS systems to boot up the virtual shell environment.
  * **Windows Core Activation:** Run `venv\Scripts\activate` on Windows platforms to bind your active command terminal.
* **Dependency Installation:**
  * **Package Syncing:** Run `pip install -r requirements.txt` to compile third-party libraries like pandas and scikit-learn into your runtime environment.
* **Credential Configuration:**
  * **Blueprint Replication:** Run `cp .env.example .env` to convert the distribution template into a live local environment configuration.
  * **Variable Customization:** Append keys like `NTFY_TOPIC=your_topic`, `SMTP_EMAIL=sender@gmail.com`, `SMTP_PASSWORD=app_pass`, and `MY_EMAIL=receiver@domain.com` directly inside the active `.env` file structure.

---

## 💻 Usage

* **Historical Data Parsing:**
  * **Batch Processing Mode:** Run `python main.py --mode batch` to compile local log exports (e.g., `~/Downloads/feed.csv`) to parse farm node variances.
* **Persistent Data Loops:**
  * **Live Continuous Monitoring:** Run `python main.py --mode live` to continuously sample live ThingSpeak channel data pools every 60 seconds in the background.
* **Stateless Network Processing:**
  * **Cron Single-Pass Evaluation:** Run `python main.py --mode cron` to fire a standalone network validation sweep, refresh active alert maps, export structural updates, and terminate.
* **Machine Learning Pipeline:**
  * **Model Optimization Engine:** Run `python main.py --mode train` to capture historical cross-node datasets, calculate timeline differences, and re-save the Isolation Forest AI models to the binaries folder.

---

## ⚙️ CI/CD Deployment

* **Automated Serverless Scheduling:**
  * **Interval Frequency:** Workflow rules inside `.github/workflows/soil_monitor.yml` fire a data check automatically **every 31 minutes** using an automated cloud timer layout (cron format).
* **Persistent Cache Management:**
  * **State Rehydration:** Runners invoke `actions/cache/restore@v4` at startup to fetch the prior iteration's `alert_state.json` cache matrix out of cloud storage.
  * **State Preservation:** Runners invoke `actions/cache/save@v4` on completion to seal the updated alert cooldown maps back into the GitHub Actions runner cache.
* **Secure Environment Masking:**
  * **Secret Cryptography:** GitHub repository secrets inject tokens like `${{ secrets.SMTP_PASSWORD }}` and `${{ secrets.NTFY_TOPIC }}` directly into hidden shell variables at runtime.
* **Evaluation Dispatching:**
  * **Core Script Execution:** The runner maps dependencies, boots a stateless Python shell, and calls `python main.py --mode cron` to perform single-pass anomaly verification.
