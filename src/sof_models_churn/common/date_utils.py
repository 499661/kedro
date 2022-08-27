import pandas as pd
from datetime import date

def convert_end_date_to_str(end_date: str) -> str:
    '''
    Convert end_date to a date string. When end_date is None convert it to a
    date string corresponding to today. 
    '''
    if end_date is not None:
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    else:
        end_date = date.today().strftime('%Y-%m-%d')
    return end_date

def convert_end_date_to_python_date(end_date: str) -> date:
    '''
    Convert end_date to a datetime.date object. When end_date is None convert it to a
    date a datetime.date object corresponding to today. 
    '''
    if end_date is not None:
        end_date = pd.to_datetime(end_date).date()
    else:
        end_date = date.today()
    return end_date

def add_colon_to_utc_offset(timestamp: str) -> str:
    '''
    Assumes that the timestamp ends with e.g. -0700 and transforms this into
    -07:00 . 
    '''
    return timestamp[:-2] + ":" + timestamp[-2:]
