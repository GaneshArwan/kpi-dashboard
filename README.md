
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44-red)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

# Corporate KPI & SLA Dashboard

## 📊 Overview

This project is a comprehensive **KPI (Key Performance Indicator) Dashboard** built with **Streamlit**. It is designed to track, analyze, and visualize the performance of internal request tickets, focusing heavily on **Service Level Agreements (SLA)**, approval turn-around times, and Master Data Management (MDM) processing efficiency.

The application integrates directly with **Google Drive and Google Sheets** to fetch live data but also includes a **Dummy Data Mode** for demonstration purposes without requiring credentials.

## 🚀 Key Features

- **📈 Interactive Analytics:**
	- **Overview:** High-level metrics on total requests, expiration rates, and overdue items.
	- **Requester Detail:** Analysis of ticket creation duration and approval bottlenecks.
	- **MDM Detail:** Deep dive into processing speeds, rejected/completed rates, and task-level efficiency.
- **⏱️ Advanced SLA Calculation:**
	- Calculates processing duration based on **business days** (excluding weekends and national holidays).
	- Custom logic for different request types (e.g., *Promotion Create* vs. *Standard Requests*).
- **🛠️ Data Handling:**
	- **Google Drive Integration:** Automates data fetching and archiving from Google Sheets.
	- **Robust ETL:** Cleans, maps types, and merges archival data with current active sheets.
	- **Dummy Data Generator:** Built-in logic to generate realistic synthetic data for testing and portfolio demos.
- **🎨 Visualization:**
	- Dynamic bar charts, line trends, box plots (for outlier detection), and pie charts using **Plotly**.

## 🛠️ Tech Stack

- **Core:** Python 3.11
- **Frontend:** Streamlit, Streamlit Shadcn UI
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Plotly Express, Plotly Graph Objects
- **Integrations:** Google API Client, gspread, OAuth2
- **Utilities:** Holidays (for SLA calculation), Dataclasses

## 📂 Project Structure

```bash
kpi-dashboard/
├── app.py                   # Main entry point (Navigation & State Management)
├── pages/
│   ├── 1_overview.py        # General trend analysis
│   ├── 2_requester_detail.py # Requester & Approver metrics
│   └── 3_mdm_detail.py      # MDM team performance & SLAs
├── utils/
│   ├── drive.py             # Google Drive API wrappers
│   ├── helper.py            # ETL, SLA Logic, and Dummy Data generation
│   └── streamlit.py         # Custom UI components & Charting functions
├── config/
│   └── dataclasses.py       # Column mapping & constants
├── requirements.txt         # Project dependencies
└── README.md
```

## 💻 Installation & Setup

1. Clone the repository

```bash
git clone https://github.com/yourusername/kpi-dashboard.git
cd kpi-dashboard
```

2. Install dependencies (use a virtual environment)

```bash
pip install -r requirements.txt
```

3. Google Cloud Credentials (Optional for Live Data)

To run with Live Data, place a `secrets.toml` file in the `.streamlit` folder with your Google Service Account credentials. If you only want the demo, skip this step and enable "Use Dummy Data" in the app sidebar.

Example `.streamlit/secrets.toml` format:

```toml
[google_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n..."
client_email = "your-email@..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
spreadsheet_id = "your-sheet-id"
drive_id = "your-drive-folder-id"
```

4. Run the Application

```bash
streamlit run app.py
```

## 🧪 Demo Mode (No Drive)

1. Launch the app.
2. In the sidebar, enable "🛠️ Use Dummy Data (No Drive)".
3. The app will generate a sample dataset (e.g., 100 synthetic rows) and populate the Overview, Requester Detail, and MDM Detail pages.

## 🧠 Business Logic Highlight

The most complex part of the dashboard is the SLA engine implemented in `utils/helper.py`:

- It determines SLA pass/fail by identifying request type (Create vs. Promo).
- Compares `Valid From` or MDM Received timestamps against dynamic deadlines.
- Adjusts deadlines for weekends and national holidays (Indonesia by default).
- Compares actual processing durations against allowed thresholds (e.g., 24h vs 48h).

## 📌 Notes & Next Steps

- To customize holidays, update the holidays logic in `utils/helper.py`.
- To connect to live Google Sheets, ensure your service account has access to the target spreadsheet and Drive folder.

---

