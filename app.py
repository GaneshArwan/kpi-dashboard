import streamlit as st
import pandas as pd
import numpy as np

from datetime import datetime
from utils.helper import (
    load_spreadsheet,
    merge_data,
    load_archive,
    get_bin_count,
    get_count,
    generate_archive,
    calculate_work_days_duration,
    calculate_kpi_pass,
    generate_dummy_data,
)
from utils.streamlit import load_multiselect, load_date_picker
from st_pages import get_nav_from_toml, add_page_title

FILTER_COLS = {
    "company_group": {
        "title": "Select Company Group",
        "column_name": "Company Group"    
    },
    "company_names": {
        "title": "Select Company Names",
        "column_name": "Company Code - Name"
    },
    "department": {
        "title": "Select Department",
        "column_name": "Department"
    },
    "request_type_group": {
        "title": "Select Request Type Group",
        "column_name": "Request Type Group"
    },
    "request_type": {
        "title": "Select Request Type",
        "column_name": "Request Type"
    },
    "mdm": {
        "title": "Select MDM",
        "column_name": "Processed By"
    }
}

def load_page():
    st.set_page_config(layout="wide")
    nav = get_nav_from_toml(".streamlit/pages_sections.toml")
    pg = st.navigation(nav)
    add_page_title(pg)
                
    load_session_state()
    load_filter()
    refresh_data()    

    #Refresh Metrics
    load_metrics()
    load_mdm_metrics()
    load_approver_metrics()
    load_requester_metrics()
    
    archive()

    pg.run()

st.cache_data()
def archive():
    if datetime.now().day == 1:
        current_date = datetime.now()
        last_month = current_date.month - 1 if current_date.month > 1 else 12
        last_month_year = current_date.year if current_date.month > 1 else current_date.year - 1

        # Filter the data for the last month
        last_month_data = st.session_state.data[
            (st.session_state.data['Timestamp'].dt.month == last_month) &
            (st.session_state.data['Timestamp'].dt.year == last_month_year)
        ]

        generate_archive(st.secrets['google_service_account'], last_month_data)
        return

@st.cache_data(ttl=604_800)          # one week
def fetch_archive(_secret_dict):     # leading "_" ⇒ argument ignored by the hasher
    return load_archive(_secret_dict)

@st.cache_data(ttl=3600)             # 1 hour cache for sheet data
def fetch_sheet(_secret_dict):
    return load_spreadsheet(_secret_dict)

# 2.  A thin, non-cached wrapper that populates session_state ---------------
def load_session_state(force=False):
    # --- Helper to switch to Dummy Data ---
    def use_dummy_mode():
        if 'dummy_data' not in st.session_state or force:
            st.session_state.dummy_data = generate_dummy_data(100)
        st.session_state.data = st.session_state.dummy_data

    # 1. If the User explicitly checked the "Dummy Mode" box, use it.
    if st.session_state.get('use_dummy', False):
        use_dummy_mode()
        return

    # 2. Otherwise, TRY to connect to Google Drive (Real Mode)
    try:
        # Check if the secrets even exist
        if 'google_service_account' not in st.secrets:
            raise ValueError("Secrets not found")

        secret = st.secrets['google_service_account']

        if 'archive' not in st.session_state:
            st.session_state.archive = fetch_archive(secret)
        if force or 'sheet' not in st.session_state:
            st.session_state.sheet = fetch_sheet(secret)

        # Merge real data
        st.session_state.data = merge_data(
            st.session_state.sheet,
            st.session_state.archive
        )

    except Exception as e:
        # 3. SAFETY NET: If Real Mode fails (no secrets, error, etc.), auto-use Dummy Mode
        print(f"⚠️ Drive Connection Failed: {e}")  # Logs error to console for you
        use_dummy_mode()

def update_cache():
    load_session_state(force=True)

def refresh_data():
    if 'data' not in st.session_state:
        load_session_state(force=True)
        return

    # work on a copy
    df = st.session_state.data.copy()

    # ── 1) Exclude 'SPECIAL PROJECT' by default ──
    if not st.session_state.get('include_special', False):
        df = df[df['Department'] != 'SPECIAL PROJECT']

    # ── 2) Dynamic filters on other cols ──
    for key, cfg in FILTER_COLS.items():
        sel = st.session_state.get(key, ['All'])

        # ───── NEW BLOCK ────────────────────────────────────────────
        # If the checkbox is ticked and the user didn't already
        # select 'All' or 'SPECIAL PROJECT', silently add it so the
        # subsequent filter doesn’t throw those rows away again.
        if key == 'department' and st.session_state.get('include_special', False):
            if 'All' not in sel and 'SPECIAL PROJECT' not in sel:
                sel = sel + ['SPECIAL PROJECT']
        # ────────────────────────────────────────────────────────────

        if 'All' not in sel:
            df = df[df[cfg['column_name']].isin(sel)]

    # ── 3) Date-range filtering (unchanged) ──
    start = st.session_state.get('start_date')
    end   = st.session_state.get('end_date')

    if start and end:
        start_ts = pd.to_datetime(start)
        end_ts   = pd.to_datetime(end)

        mask1 = (df['Timestamp'] >= start_ts) & (df['Timestamp'] <= end_ts)
        mask2 = (df['Timestamp MDM Received'] >= start_ts) & (df['Timestamp MDM Received'] <= end_ts)

        st.session_state.filtered_data          = df.loc[mask1]
        st.session_state.filtered_data_approved = df.loc[mask2]
    else:
        st.session_state.filtered_data          = df
        st.session_state.filtered_data_approved = df
        
def load_filter():
    # --- Date Picker ---
    use_dummy = st.sidebar.checkbox("🛠️ Use Dummy Data (No Drive)", value=False)
    st.session_state.use_dummy = use_dummy
    
    if st.session_state.use_dummy:
        st.sidebar.warning("⚠️ Running in Test Mode")
    # ----------------------

    # ... rest of your existing load_filter code ...
    # start_date, end_date = load_date_picker() ...
    start_date, end_date = load_date_picker()
    st.session_state.start_date = start_date
    st.session_state.end_date   = end_date

    # --- Multiselect Filters ---
    if 'data' not in st.session_state:
        load_session_state(force=True)

    data = st.session_state.data
    for key, value in FILTER_COLS.items():
        selected_values = load_multiselect(
            title=value['title'],
            data=data[value['column_name']]
        )
        st.session_state[key] = selected_values if selected_values else ['All']

    # --- Include SPECIAL PROJECT Checkbox ---
    include_special = st.sidebar.checkbox(
        "Include SPECIAL PROJECT",
        value=False
    )
    st.session_state.include_special = include_special

    st.sidebar.button("⟳ Refresh Data", on_click=update_cache)

# --- LOAD METRICS ---
def load_metrics():
    filtered_data = st.session_state.filtered_data
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    
    st.session_state.total_requests = len(filtered_data)
    st.session_state.unique_requests = filtered_data['NIP'].nunique()
    
    days_in_range = (end_date - start_date).days + 1
    st.session_state.avg_requests_per_day = st.session_state.total_requests / days_in_range if days_in_range > 0 else 0

    # Optimize count operations with vectorized operations
    requester_counts = filtered_data['Respon Requester'].value_counts()
    expired_count = requester_counts.get('Expired', 0)
    completed_count = requester_counts.get('Completed', 0)
    total_response = completed_count + expired_count
    st.session_state.completion_rate_percentage = (completed_count / total_response) * 100 if total_response > 0 else 0 
    st.session_state.expired_count = expired_count
    st.session_state.completed_count = completed_count

    # Optimize approver count operations
    approver_counts = filtered_data['Respon Approver'].value_counts()
    no_approver_count = approver_counts.get('NO APPROVER', 0)
    approved_count = approver_counts.get('Approved', 0)
    rejected_count = approver_counts.get('Rejected', 0)
    
    total_approver_response = approved_count + no_approver_count + rejected_count
    st.session_state.approval_rate_percentage = (approved_count / total_approver_response) * 100 if total_approver_response > 0 else 0
    st.session_state.no_approver_count = no_approver_count
    st.session_state.approved_count = approved_count
    st.session_state.rejected_count = rejected_count

def load_mdm_metrics():    
    # 1) Make a copy to avoid SettingWithCopyWarning
    filtered_data = st.session_state.filtered_data.copy()

    # 2) compute your “overdue” subset as before
    st.session_state.filtered_data_overdue = filtered_data[
        filtered_data['Timestamp MDM Received'].notna() &
        filtered_data['Process Status'].isna() &
        (pd.to_numeric(filtered_data['Overdue Duration (Minutes)'], errors='coerce') > 1440)
    ]

    # 3) drop truly “blank” process-status rows
    filtered_data = filtered_data.dropna(subset=['Process Status'])

    # 4) Optimize work-days calculation with vectorized operations where possible
    # Create boolean masks first
    has_mdm_received = filtered_data['Timestamp MDM Received'].notna()
    has_processed_date = filtered_data['Processed Date'].notna()
    needs_duration_calc = has_mdm_received & has_processed_date
    
    # Initialize the column
    filtered_data['Processed Duration (Minutes)'] = np.nan
    
    # Only calculate for rows that have both timestamps
    if needs_duration_calc.any():
        filtered_data.loc[needs_duration_calc, 'Processed Duration (Minutes)'] = filtered_data.loc[needs_duration_calc].apply(
            lambda row: calculate_work_days_duration(
                row['Timestamp MDM Received'],
                row['Processed Date']
            ),
            axis=1
        )

    # 5) per-task timing - ensure both columns are numeric
    filtered_data['Processed Duration (Minutes)'] = pd.to_numeric(
        filtered_data['Processed Duration (Minutes)'],
        errors='coerce'
    )
    filtered_data['Total Task'] = (
        pd.to_numeric(filtered_data['Total Task'], errors='coerce')
        .fillna(1)
        .astype(int)
        .replace(0, 1)
    )

    filtered_data['Processed Duration per Task'] = (
        filtered_data['Processed Duration (Minutes)'] /
        filtered_data['Total Task']
    ).round(2)

    # 6) **NEW**: capture errors for debugging
    mdm_error_mask = filtered_data['Processed Duration (Minutes)'].isna()
    st.session_state.mdm_error_rows = filtered_data[mdm_error_mask].copy()

    # 7) **NEW**: compute days only on the valid rows
    valid = ~mdm_error_mask
    filtered_data.loc[valid, 'Processed Duration (Day)'] = (
        np.ceil(filtered_data.loc[valid, 'Processed Duration (Minutes)'] / 1440)
        .astype('Int64')
    )
    # ensure the invalid ones show up as <NA>
    filtered_data.loc[mdm_error_mask, 'Processed Duration (Day)'] = pd.NA
    filtered_data['Processed Duration (Day)'] = filtered_data['Processed Duration (Day)'].astype('Int64')

    # 8) KPI calculation using business-day rules
    filtered_data['KPI Pass'] = calculate_kpi_pass(filtered_data)


    # 9) Optimize summary stats with vectorized operations
    st.session_state.mdm_average_processed_per_task = (
        pd.to_numeric(filtered_data['Processed Duration per Task'], errors='coerce')
        .mean() / 60
    )
    st.session_state.mdm_average_processed_duration = (
        filtered_data['Processed Duration (Minutes)'].mean() / 60
    )
    st.session_state.mdm_processed_duration_count = get_bin_count(
        filtered_data, 'Processed Duration (Minutes)'
    )
    
    # Optimize process status counts with single value_counts call
    process_status_counts = filtered_data['Process Status'].value_counts()
    st.session_state.mdm_rejected_count = process_status_counts.get('Rejected', 0)
    st.session_state.mdm_partial_rejected_count = process_status_counts.get('Partially Rejected', 0)
    st.session_state.mdm_completed_count = process_status_counts.get('Completed', 0)
    st.session_state.mdm_send_back_count = process_status_counts.get('Send Back', 0)
    
    # Ensure Total Task is numeric before summing
    st.session_state.mdm_total_task = pd.to_numeric(filtered_data['Total Task'], errors='coerce').sum()

    total_mdm_responses = (
        st.session_state.mdm_completed_count +
        st.session_state.mdm_partial_rejected_count +
        st.session_state.mdm_rejected_count + 
        st.session_state.mdm_send_back_count
    )
    st.session_state.mdm_completed_rate_percentage = (
        (st.session_state.mdm_completed_count / total_mdm_responses) * 100
        if total_mdm_responses > 0 else 0
    )
    st.session_state.mdm_rejected_rate_percentage = (
        (st.session_state.mdm_rejected_count / total_mdm_responses) * 100
        if total_mdm_responses > 0 else 0
    )
    st.session_state.mdm_partially_rejected_rate_percentage = (
        (st.session_state.mdm_partial_rejected_count / total_mdm_responses) * 100
        if total_mdm_responses > 0 else 0
    )
    st.session_state.mdm_send_back_rate_percentage = (
        (st.session_state.mdm_send_back_count / total_mdm_responses) * 100
        if total_mdm_responses > 0 else 0
    )

    st.session_state.mdm_avg_task_per_request = (
        st.session_state.mdm_total_task / total_mdm_responses
        if total_mdm_responses > 0 else 0
    )

    # 10) store the “clean” DF back
    st.session_state.mdm_filtered_data = filtered_data


def load_requester_metrics():
    # Make a copy to avoid SettingWithCopyWarning
    filtered_data = st.session_state.filtered_data.copy()
    # Filter once for completed requests
    completed_data = filtered_data[filtered_data['Respon Requester'] == 'Completed'].copy()
    
    if not completed_data.empty:
        # Vectorized duration calculation
        completed_data['Creation Duration (Minutes)'] = (
            completed_data['Timestamp Requester'] - completed_data['Timestamp']
        ).dt.total_seconds() / 60
        
        st.session_state.requester_avg_creation_duration = pd.to_numeric(
            completed_data['Creation Duration (Minutes)'], errors='coerce'
        ).mean()
        st.session_state.requester_creation_duration_count = get_bin_count(
            completed_data, 'Creation Duration (Minutes)'
        )
    else:
        st.session_state.requester_avg_creation_duration = 0
        st.session_state.requester_creation_duration_count = pd.DataFrame()
    
    st.session_state.requester_filtered_data = completed_data
    
def load_approver_metrics():
    # Make a copy to avoid SettingWithCopyWarning
    filtered_data = st.session_state.filtered_data.copy()
    # Filter once for approved requests
    approved_data = filtered_data[filtered_data['Respon Approver'] == 'Approved'].copy()
    
    if not approved_data.empty:
        # Vectorized duration calculation
        approved_data['Approval Duration (Minutes)'] = (
            approved_data['Timestamp MDM Received'] - approved_data['Timestamp Requester']
        ).dt.total_seconds() / 60
        
        st.session_state.approver_avg_approval_duration = pd.to_numeric(
            approved_data['Approval Duration (Minutes)'], errors='coerce'
        ).mean()
        st.session_state.approver_duration_count = get_bin_count(
            approved_data, 'Approval Duration (Minutes)'
        )
    else:
        st.session_state.approver_avg_approval_duration = 0
        st.session_state.approver_duration_count = pd.DataFrame()
    
    st.session_state.approver_filtered_data = approved_data
    
    
if __name__ == "__main__":
    load_page()
