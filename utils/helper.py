from gspread_dataframe import get_as_dataframe
from utils.drive import (
    get_credentials, 
    get_spreadsheet, 
    upload_dataframe_to_drive,
    load_df_files_from_drive
)
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import calendar
import holidays
import streamlit as st

HOLIDAY_CALENDAR = holidays.country_holidays('ID')

COLUMNS_MAPPING = {
    'Email Address': 'string',
    'Request Number': 'string',
    'Timestamp': 'datetime64[ns]',
    'NIP': 'string',
    'Company Code - Name': 'string',
    'Department': 'string',
    'Request Type': 'string',
    'Promo Type': 'string',
    'Respon Requester': 'string',
    'Timestamp Requester': 'datetime64[ns]',
    'Name Requester': 'string',
    'Respon Approver': 'string',
    'Timestamp Approver': 'datetime64[ns]',
    'Name Approver': 'string',
    'Respon Approver II': 'string',
    'Timestamp Approver II': 'datetime64[ns]',
    'Name Approver II': 'string',
    'Respon Approver III': 'string',
    'Timestamp Approver III': 'datetime64[ns]',
    'Name Approver III': 'string',
    'Processed By': 'string',
    'Process Status': 'string',
    'Taken Date': 'datetime64[ns]',
    'Processed Date': 'datetime64[ns]',
    'Request Type Group': 'string',
    'Total Task': 'int',
    'Company Group': 'string',
    'Timestamp MDM Received': 'datetime64[ns]',
    'Valid From': 'datetime64[ns]',
    'Valid To': 'datetime64[ns]',
    'Attachment': 'string'
}

COMPANY_GROUP_MAP = {
    'I': 'Industrial',
    'R': 'Retail',
    'F': 'Food and Beverages',
    'S': 'Service',
    'M': 'Manufacture',
    'P': 'Property',
    'A': 'Agriculture',
    'K': 'Koperasi',
}

SHEET_ACTIVITIES = [
    'EXTEND PIR', 'HIERARCHY', 'BOM', 'NON M', 'BASIC DATA',
    'SOURCE LIST', 'IMAGE', 'MERCHANDISE', 'STATUS/LISTING', 'MASTER DATA', 'PROMOTION',
    'MASTER SITE', 'MASTER FINANCE', 'PRICING', 'PROFIT CENTER', 'VENDOR', 'LISTING'
]

def load_archive(secret_key):
    creds = get_credentials(secret_key)
    return load_df_files_from_drive(creds)


def load_spreadsheet(secret_key):
    creds       = get_credentials(secret_key)
    spreadsheet = get_spreadsheet(creds)

    # Pre-allocate list for better performance
    dataframes = []
    columns     = list(COLUMNS_MAPPING.keys())

    for sheet in spreadsheet.worksheets():
        if sheet.title not in SHEET_ACTIVITIES:
            continue
        ws = spreadsheet.worksheet(sheet.title)
        df = get_as_dataframe(ws, evaluate_formulas=True, skiprows=1)
        if df.empty:
            continue

        # Vectorized operations
        df['Request Type Group'] = sheet.title
        df['Company Group'] = df['Company Code - Name'].str[0].map(COMPANY_GROUP_MAP)

        # Optimized rejection filtering
        respon_cols = [col for col in df.columns if 'Respon' in col]
        if respon_cols:
            rej = df[respon_cols].isin(['Rejected', 'Expired']).any(axis=1)
        else:
            rej = pd.Series([False] * len(df), index=df.index)
            
        # Find timestamp column more efficiently
        ts_cols = ['Timestamp Approver III', 'Timestamp Approver II', 'Timestamp Approver', 'Timestamp Requester']
        ts_col = None
        for col in ts_cols:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col:
            df['Timestamp MDM Received'] = df[ts_col].where(~rej)
        else:
            df['Timestamp MDM Received'] = pd.NaT

        # Apply column mapping before filtering columns
        df = handle_col_map(df)
        
        # Only select needed columns
        available_columns = [col for col in columns if col in df.columns]
        dataframes.append(df[available_columns])

    # Concatenate all dataframes at once for better performance
    if dataframes:
        data_master = pd.concat(dataframes, ignore_index=True, sort=False)
    else:
        data_master = pd.DataFrame(columns=columns)

    return data_master


def merge_data(sheet_df: pd.DataFrame, archive_df: pd.DataFrame) -> pd.DataFrame:
    # Concatenate dataframes more efficiently
    data = pd.concat([sheet_df, archive_df], ignore_index=True, sort=False)
    
    # Drop duplicates more efficiently  
    data.drop_duplicates(subset=['Request Number'], keep='first', inplace=True)
    data = handle_col_map(data)
    data = data.fillna({'Department': 'All'})

    # Optimize the overdue duration calculation
    now = pd.Timestamp.now()
    
    # Create boolean masks for better performance
    has_mdm_received = data['Timestamp MDM Received'].notna()
    no_processed_date = data['Processed Date'].isna()
    needs_calculation = has_mdm_received & no_processed_date
    
    # Only calculate for rows that need it
    if needs_calculation.any():
        data.loc[needs_calculation, 'Overdue Duration (Minutes)'] = data.loc[needs_calculation].apply(
            lambda row: calculate_work_days_duration(row['Timestamp MDM Received'], now),
            axis=1
        )
    
    # Set other rows to NaN
    data.loc[~needs_calculation, 'Overdue Duration (Minutes)'] = np.nan

     # ── NEW: normalize “Processed By” to have uppercase first letter ──
    # strip whitespace, lowercase everything, then capitalize first char
    data['Processed By'] = (
        data['Processed By']
        .astype(str)
        .str.strip()
        .str.lower()
        .str.capitalize()
    )
    
    return data

def generate_archive(secret_key, dataframe):
    last_month = datetime.now() - timedelta(days=30)
    if last_month.month == datetime.now().month:
        last_month = last_month.replace(day=1) - timedelta(days=1)
    filename = f"{last_month.year}_{last_month.month}_Activities"
    
    upload_dataframe_to_drive(get_credentials(secret_key), dataframe, filename)

# --- DATAFRAME HELPER ---

def handle_col_map(df):
    # Strip column names once
    df.columns = df.columns.str.strip()
    
    # Separate datetime, numeric, and other columns for optimized batch processing
    datetime_columns = []
    numeric_columns = []
    other_columns = []
    
    for column, dtype in COLUMNS_MAPPING.items():
        if column not in df.columns:
            df[column] = pd.Series(np.nan, dtype=dtype)
        else:
            if 'datetime' in dtype:
                datetime_columns.append(column)
            elif 'int' in dtype or 'float' in dtype:
                numeric_columns.append((column, dtype))
            else:
                other_columns.append((column, dtype))
    
    # Batch process datetime columns
    for column in datetime_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors='coerce')
    
    # Process numeric columns with better error handling
    for column, dtype in numeric_columns:
        if column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
                # Cast to specific numeric type if needed
                if 'int' in dtype:
                    df[column] = df[column].astype('Int64')  # Nullable integer
            except Exception:
                # If conversion fails, keep original data
                pass
    
    # Process other columns
    for column, dtype in other_columns:
        if column in df.columns:
            try:
                df[column] = df[column].astype(dtype, errors='ignore')
            except Exception:
                # If conversion fails, keep original data
                pass
            
    return df

def get_count(data, column, value=None):
    if column not in data.columns:
        raise ValueError('Column not exist in DataFrame')
    
    if value:
        return data[data[column] == value].shape[0]
    
    return data[column].shape[0]
    

def get_column_counts(data: pd.DataFrame, column_name: str, count_column_name: str = 'Count') -> pd.DataFrame:
    """
    Returns a DataFrame containing the counts of unique values in the specified column.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column for which to count unique values.
    - count_column_name (str): The name of the column to store the counts. Default is 'Count'.

    Returns:
    - pd.DataFrame: A DataFrame with the unique values and their counts.
    """
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    
    return data[column_name].value_counts().sort_index()

def get_unique_month_years(start_date_str, end_date_str):
    # Parse the start and end dates
    start_date = datetime.strptime(start_date_str, '%d-%m-%Y')
    end_date = datetime.strptime(end_date_str, '%d-%m-%Y')
    
    # List to store unique (month, year) tuples
    unique_month_years = set()
    
    # Iterate through each month in the range
    current_date = start_date
    while current_date <= end_date:
        # Add the month's name and year to the set
        month_name = calendar.month_name[current_date.month]
        year = current_date.year
        unique_month_years.add(f'{month_name} {year}')
        
        # Move to the next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return sorted(
        unique_month_years, 
        key=lambda x: (int(x.split()[1]), 
                       list(calendar.month_name).index(x.split()[0])))


 # --- OTHER ---

def wrap_label(    
    label, 
    max_length=15
    ):
    
    """ Multi-line text"""
    words = label.split()  # Split label into words
    lines = []
    current_line = ""

    for word in words:
        # Check if adding this word exceeds the max length
        if len(current_line) + len(word) + 1 > max_length:
            lines.append(current_line.strip())
            current_line = word
        else:
            if current_line:  # Add a space between words
                current_line += " "
            current_line += word

    # Add the last line if there's any content
    if current_line:
        lines.append(current_line.strip())

    return '<br>'.join(lines)

def get_date_range(selection):
    today = datetime.now()

    date_ranges = {
        "Today": (
            today.replace(hour=0, minute=0, second=1, microsecond=0),
            today.replace(hour=23, minute=59, second=59, microsecond=0)
        ),
        "Yesterday": (
            (today - timedelta(days=1)).replace(hour=0, minute=0, second=1, microsecond=0),
            (today - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
        ),
        "This Week": (
            (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=1, microsecond=0),
            (today - timedelta(days=today.weekday()) + timedelta(days=6)).replace(hour=23, minute=59, second=59, microsecond=0)
        ),
        "Last 30 Days": (
            (today - timedelta(days=30)).replace(hour=0, minute=0, second=1, microsecond=0),
            today.replace(hour=23, minute=59, second=59, microsecond=0)
        ),
        "Custom Date": (
            (today - timedelta(days=30)).replace(hour=0, minute=0, second=1, microsecond=0),
            today.replace(hour=23, minute=59, second=59, microsecond=0)
        )
    }

    return date_ranges.get(selection)


# --- DATAFRAME HELPER ---

def calculate_work_days_duration(start, end):
    """
    Calculate duration excluding weekends and holidays.
    Optimized version with early returns and better logic.
    """
    if pd.isnull(start) or pd.isnull(end):
        return np.nan
        
    # Convert to datetime if not already
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    # Early return if same day
    if start.date() == end.date():
        if start.weekday() < 5 and start.date() not in HOLIDAY_CALENDAR:
            return round((end - start).total_seconds() / 60, 2)
        else:
            return 0.0
    
    # Early return if end is before start
    if end <= start:
        return 0.0
    
    # Initialize duration tracking
    total_duration_minutes = 0
    current = start
    
    # Limit loop iterations for very long periods
    max_days = 365  # Prevent infinite loops for extreme date ranges
    days_processed = 0
    
    while current < end and days_processed < max_days:
        # Check if current day is a workday (not weekend or holiday)
        if current.weekday() < 5 and current.date() not in HOLIDAY_CALENDAR:
            # Calculate duration for the current day
            next_day = pd.Timestamp(current.date() + timedelta(days=1))
            day_end = min(end, next_day)
            
            # Add the duration for the current day
            if current < day_end:
                day_duration = (day_end - current).total_seconds() / 60
                total_duration_minutes += day_duration
        
        # Move to the start of the next day
        current = pd.Timestamp(current.date() + timedelta(days=1))
        days_processed += 1
    
    return round(total_duration_minutes, 2)


def calculate_kpi_pass(df: pd.DataFrame) -> pd.Series:
    """
    Calculate KPI pass flag using business-day rules for each request type.
    """
    if df.empty:
        return pd.Series(dtype=bool)

    request_type = df["Request Type"].fillna("")
    processed_date = pd.to_datetime(df["Processed Date"], errors="coerce")
    valid_from = pd.to_datetime(df["Valid From"], errors="coerce")
    mdm_received = pd.to_datetime(df["Timestamp MDM Received"], errors="coerce")
    dur = pd.to_numeric(df["Processed Duration (Minutes)"], errors="coerce")

    is_create = request_type.str.contains("Create Article Merchandise", na=False)
    is_promo = request_type.eq("Promotion Create")
    is_other = ~is_create & ~is_promo

    # --- Promotion: SLA based on Valid From business day ---
    vf_wday = valid_from.dt.weekday
    vf_roll_add = np.select([vf_wday == 5, vf_wday == 6], [2, 1], default=0)
    vf_rolled_to_bd = valid_from + pd.to_timedelta(vf_roll_add, unit="D")
    vf_next_add = np.select([vf_wday == 4, vf_wday == 5, vf_wday == 6], [3, 2, 1], default=1)
    vf_next_bd = valid_from + pd.to_timedelta(vf_next_add, unit="D")

    promo_has_vf = valid_from.notna()
    promo_has_mdm = mdm_received.notna()

    promo_shift_mask = (
        is_promo & promo_has_vf & promo_has_mdm &
        (mdm_received >= valid_from)
    )
    promo_deadline = vf_rolled_to_bd.where(~promo_shift_mask, vf_next_bd)

    promo_date_based_pass = (
        is_promo &
        processed_date.notna() &
        promo_deadline.notna() &
        (processed_date.dt.date <= promo_deadline.dt.date)
    )
    promo_same_day_pass = (
        is_promo &
        processed_date.notna() &
        mdm_received.notna() &
        (processed_date.dt.date == mdm_received.dt.date)
    )
    promo_pass = promo_date_based_pass | promo_same_day_pass

    # --- Other request types: next business day from MDM Received ---
    mdm_has = mdm_received.notna()
    mdm_wday = mdm_received.dt.weekday
    mdm_next_add = np.select(
        [mdm_wday == 4, mdm_wday == 5, mdm_wday == 6],
        [3, 2, 1],
        default=1
    )
    mdm_next_bd = mdm_received + pd.to_timedelta(mdm_next_add, unit="D")

    other_same_day_pass = (
        is_other & mdm_has &
        processed_date.notna() &
        (processed_date.dt.date == mdm_received.dt.date)
    )
    other_bday_pass = (
        is_other & mdm_has &
        processed_date.notna() &
        (processed_date.dt.date <= mdm_next_bd.dt.date)
    )
    other_fallback_pass = (
        is_other & ~mdm_has &
        (dur < 1440)
    )
    other_pass = other_same_day_pass | other_bday_pass | other_fallback_pass

    # --- Create requests: 48 working hours worth of minutes ---
    create_pass = is_create & (dur < 2880)

    return create_pass | promo_pass | other_pass


def handle_remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def get_column_counts(data: pd.DataFrame, column_name: str, count_column_name: str = 'Count') -> pd.DataFrame:
    """
    Returns a DataFrame containing the counts of unique values in the specified column.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column for which to count unique values.
    - count_column_name (str): The name of the column to store the counts. Default is 'Count'.

    Returns:
    - pd.DataFrame: A DataFrame with the unique values and their counts.
    """
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    
    return data[column_name].value_counts().sort_index()

def get_column_duration():
    pass

def get_bin_count(data, duration_col):
    """
    Optimized bin counting with better performance for large datasets.
    """
    if data.empty or duration_col not in data.columns:
        return pd.DataFrame()
    
    # Remove NaN values before binning for better performance
    clean_data = data.dropna(subset=[duration_col])
    
    if clean_data.empty:
        return pd.DataFrame()
    
    bins = [0, 5, 10, 15, 30, 60, 120, 180, 240, 300, 1440, 2880, 4320, 5760, 7200]
    labels = ['0-5 min', '5-10 min', '10-15 min', '15-30 min', '30-60 min', '1-2 hr', '2-3 hr', '3-4 hr', '4-5 hr', '5-24 hr', '1-2 days', '2-3 days', '3-4 days', '4-5 days']
    
    # Use pd.cut directly on the series for better performance
    duration_bins = pd.cut(clean_data[duration_col],
                          bins=bins,
                          labels=labels,
                          right=False,
                          include_lowest=True)
    
    return duration_bins.value_counts().sort_index()

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    # 1) pivot/count each status (using “Send Back” exactly)
    statuses = ["Completed", "Rejected", "Send Back", "Partially Rejected"]
    status_counts = (
        df
        .groupby(["Processed By", "Process Status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=statuses, fill_value=0)
    )

    # 2) total requests = sum of those four buckets
    status_counts["Total"] = status_counts.sum(axis=1)

    # 3) total task
    total_task = df.groupby("Processed By")["Total Task"].sum().rename("Total Task")

    # 4) on-time execution rate = mean of KPI Pass
    ontime = df.groupby("Processed By")["KPI Pass"].mean().rename("On-Time Execution Rate")

    # 5) completion rate = (Completed + Partially Rejected) / Total
    completion = (
        (status_counts["Completed"] + status_counts["Partially Rejected"])
        / status_counts["Total"]
    ).rename("Completion Rate")

    # 6) assemble and reset index
    summary = pd.concat([status_counts, total_task, ontime, completion], axis=1)
    summary = summary.reset_index().rename(columns={"Processed By": "Name"})

    # 7) final column order with “Send Back”
    summary = summary[
        [
            "Name",
            "Total",
            "Completed",
            "Rejected",
            "Send Back",
            "Partially Rejected",
            "Total Task",
            "On-Time Execution Rate",
            "Completion Rate",
        ]
    ]

    # 8) formatting
    for c in ["Total", "Completed", "Rejected", "Send Back", "Partially Rejected", "Total Task"]:
        summary[c] = summary[c].map("{:,.0f}".format)

    summary["On-Time Execution Rate"] = summary["On-Time Execution Rate"].map("{:.2%}".format)
    summary["Completion Rate"]          = summary["Completion Rate"].map("{:.2%}".format)

    return summary
import random

def generate_dummy_data(num_rows=100):
    import random
    from datetime import datetime, timedelta
    
    # 1. Setup Data Pools
    companies = ['1001 - Alpha Corp', '1002 - Beta Ltd', '2001 - Gamma Inc', '3000 - Delta Group']
    departments = ['HR', 'Finance', 'IT', 'Marketing', 'Operation', 'SPECIAL PROJECT']
    req_types = ['Create Article Merchandise', 'Promotion Create', 'Vendor Registration', 'Master Data Update']
    # Include 'None' to simulate Open/Pending tickets
    statuses = ['Completed', 'Rejected', 'Partially Rejected', 'Send Back', None] 
    mdms = ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Brown']
    
    data = []
    now = datetime.now()

    for i in range(num_rows):
        # Randomize timing
        base_time = now - timedelta(days=random.randint(0, 30))
        mdm_recv = base_time + timedelta(hours=random.randint(1, 24))
        
        # Determine Status
        status = random.choice(statuses)
        
        # If status is None (Open Ticket), Processed Date is None
        if status is None:
            proc_date = pd.NaT
            # Calculate Overdue Duration if it's still open
            # (Simple logic: Time since MDM Received until Now)
            overdue_min = (now - mdm_recv).total_seconds() / 60
        else:
            proc_date = mdm_recv + timedelta(hours=random.randint(1, 48))
            overdue_min = 0 # Closed tickets aren't "overdue" in this context

        row = {
            'Request Number': f"REQ-{1000+i}",
            'Timestamp': base_time,
            'Email Address': f"user{i}@example.com",
            'NIP': f"NIP-{100+i}",
            'Company Code - Name': random.choice(companies),
            'Department': random.choice(departments),
            'Request Type': random.choice(req_types),
            'Promo Type': 'Regular',
            
            # Requester Flow
            'Respon Requester': 'Completed',
            'Timestamp Requester': base_time + timedelta(minutes=10),
            'Name Requester': f"Requester {i}",
            'Respon Approver': 'Approved',
            'Timestamp Approver': base_time + timedelta(hours=2),
            'Name Approver': f"Approver {i}",
            
            # MDM Flow
            'Timestamp MDM Received': mdm_recv,
            'Processed By': random.choice(mdms),
            'Process Status': status,
            'Taken Date': mdm_recv + timedelta(minutes=random.randint(5, 60)),
            'Processed Date': proc_date,
            
            # CRITICAL FIX: Add the missing column
            'Overdue Duration (Minutes)': overdue_min, 
            
            # Metadata
            'Request Type Group': 'Master Data',
            'Total Task': random.randint(1, 10),
            'Company Group': random.choice(['Industrial', 'Retail', 'Service']),
            'Valid From': base_time,
            'Valid To': base_time + timedelta(days=30),
            'Attachment': 'dummy_link'
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Ensure types are correct, but keep our new column
    df = handle_col_map(df)
    
    # handle_col_map might drop columns it doesn't know, so we re-assign if lost, 
    # OR we just ensure it persists. 
    # To be safe, let's force the column existence after mapping:
    if 'Overdue Duration (Minutes)' not in df.columns:
         # Map it from the original list if handle_col_map dropped it
         df['Overdue Duration (Minutes)'] = [d['Overdue Duration (Minutes)'] for d in data]

    return df