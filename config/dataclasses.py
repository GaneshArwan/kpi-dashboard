from dataclasses import dataclass
from typing import Dict

@dataclass
class ColumnConfig:
    """Configuration for DataFrame columns and their types"""
    COLUMNS_MAPPING: Dict[str, str] = {
        'Request Number': 'string',
        'Timestamp': 'datetime64[ns]',
        'NIP': 'string',
        'Company Code - Name': 'string',
        'Department': 'string',
        'Request Type': 'string',
        'Respon Requester': 'string',
        'Promo Type': 'string',
        'Timestamp Requester': 'datetime64[ns]',
        'Name Requester': 'string',
        'Respon Approver': 'string',
        'Timestamp Approver': 'datetime64[ns]',
        'Name Approver': 'string',
        'Processed By': 'string',
        'Process Status': 'string',
        'Taken Date': 'datetime64[ns]',
        'Processed Date': 'datetime64[ns]',
        'Request Type Group': 'string',
        'Total Task': 'int',
        'Company Group': 'string'
    }

    COMPANY_GROUP_MAP: Dict[str, str] = {
        'I': 'Industrial',
        'R': 'Retail',
        'F': 'Food and Beverages',
        'S': 'Service',
        'M': 'Manufacture',
        'P': 'Property',
    }
    
class FilterConfig:
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
    
    
    
