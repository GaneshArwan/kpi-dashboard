import gspread
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from googleapiclient.discovery import build
from io import BytesIO


SPREADSHEET_ID = st.secrets.get("spreadsheet_id", "YOUR_SPREADSHEET_ID_HERE")
DRIVE_ID = st.secrets.get("drive_id", "YOUR_DRIVE_ID_HERE")

def get_spreadsheet(credentials, spreadsheet_id=SPREADSHEET_ID):
    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(spreadsheet_id)
    
    return spreadsheet

def get_credentials(secret_key):
    scope = ['https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_info(
        secret_key, scopes=scope)
    return credentials

def upload_dataframe_to_drive(
    credentials,
    dataframe,
    file_name,
    drive_id=DRIVE_ID
):
    service = build('drive', 'v3', credentials=credentials)
    
    # Check if file already exists
    query = f"'{drive_id}' in parents and name='{file_name}'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    
    if not files:  # If file does not exist, proceed with upload
        file_metadata = {
            'name': file_name,
            'mimeType': 'text/csv'
        }
        
        if drive_id:
            file_metadata['parents'] = [drive_id]
        
        # Convert DataFrame to CSV and store in BytesIO buffer
        buffer = BytesIO()
        dataframe.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)
        
        # Create media upload object
        media = MediaIoBaseUpload(buffer, mimetype='text/csv', resumable=True)
        
        # Upload file
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file.get("id")
    return None

def load_df_files_from_drive(credentials, drive_id=DRIVE_ID):

    service = build('drive', 'v3', credentials=credentials)

    

    # Query to list all CSV files in the specified drive folder

    query = f"'{drive_id}' in parents and mimeType='text/csv'"

    results = service.files().list(q=query, fields="files(id, name)").execute()

    files = results.get('files', [])



    dataframes = []  # List to hold individual DataFrames

    all_columns = set()  # Collects all unique column names



    for file in files:

        file_id = file['id']

        

        # Download the CSV file

        request = service.files().get_media(fileId=file_id)

        buffer = BytesIO()

        downloader = MediaIoBaseDownload(buffer, request)



        done = False

        while not done:

            status, done = downloader.next_chunk()

        

        buffer.seek(0)

        df = pd.read_csv(buffer)



        all_columns.update(df.columns)  # Collect all column names

        dataframes.append(df)



    # If no dataframes to concatenate, return an empty dataframe

    if not dataframes:

        return None



    # Ensure all DataFrames have the same column order

    all_columns = sorted(all_columns)  # Maintain consistent order

    standardized_dataframes = [df.reindex(columns=all_columns, fill_value="") for df in dataframes]


    # Concatenate all DataFrames

    concatenated_dataframe = pd.concat(standardized_dataframes, ignore_index=True)

    

    return concatenated_dataframe
