
from json import dumps
from json import loads
import numpy as np
import pandas as pd
from pandas import Index, Timestamp
#from .classes import TimestampError
from datetime import datetime, timedelta
import sys

# Functions -------------------------------------------------------------------------------------------------------------------------

"""
Calculates the sea level pressure using observed station pressure and the station elevation (in meters).
Returns a float rounded to one decimal place to be added into the dataframe.
-----
df -- the dataframe for which to calculate sea level pressure
alt -- the altitude of the station in meters
-----

Sea Level Pressure Approximation:
    SLP = P x (1 + (alt/44330))^5.255                                    
Where:
    • P is the observed atmospheric pressure at the given altitutde.
    • alt is the altitude above sea level in meters
    • This constant is derived from the barometric formula and represents a scale height in meters. 
       It is based on an average temperature and pressure profile of the atmosphere.
    • This exponent is derived from the standard atmosphere model, which assumes a constant lapse rate 
       and specific gas constant for dry air. It accounts for how pressure changes with altitude in a non-linear fashion.
"""
def calc_slp(df:pd.DataFrame, alt:int) -> float:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The calc_slp() function expects 'df' parameter to be of type <DataFrame>, passed: {type(df)}")
    if not isinstance(alt, int):
        raise TypeError(f"The calc_slp() function expects 'alt' parameter to be of type <int>, passed: {type(alt)}")
    
    return round(df["bmp2_pres"]*(293/(293-(alt*0.0065))), 1)


"""
Helper function to fill_empty_rows()
Builds a row to be merged with the dataframe for a missing timestamp.
Uses the dataframe's column order.

columns -- the df's column attribute
missing_timestamp -- the timestamp for the missing row
fill_empty -- [OPTIONAL] specify what value should populate cells, np.nan by default 
"""
def _build_empty_row(columns:Index, missing_timestamp:Timestamp, fill_empty=np.nan) -> list:
    if not isinstance(columns, Index):
        raise TypeError(f"The build_empty_row() function expects 'columns' parameter to be of type <Index>, passed: {type(columns)}")
    if not isinstance(missing_timestamp, Timestamp):
        raise TypeError(f"The build_empty_row() function expects 'missing_timestamp' parameter to be of type <Timestamp>, passed: {type(missing_timestamp)}")
    
    if pd.isna(missing_timestamp) or pd.isnull(missing_timestamp): return None

    next_row = []

    for col in columns:
        if col == 'date': next_row.append(missing_timestamp)
        else: next_row.append(fill_empty)

    return next_row


"""
Create a list to merge with the reformatted dataframe that fills in data gaps with missing timestamps. 
Merges that list with the reformatted dataframe, sorts by time, and returns it. 
REQUIREMENT: the timestamp column must be named 'date' ---------------------------------------------------

reformatted_df -- the df to reformat with data gaps filled
time_delta -- the dataset's resolution (1-minute reporting period)
set_index -- specify whether you want the dataframe returned by fc'n to have the index set (default False)
"""
def fill_empty_rows(reformatted_df:pd.DataFrame, time_delta:timedelta, set_index:bool=False):
    if not isinstance(reformatted_df, pd.DataFrame):
        raise ValueError(f"The fill_empty_row() function expects the 'reformatted_df' parameter to be of type <pd.DataFrame>, received: {type(reformatted_df)}")
    if not isinstance(time_delta, timedelta):
        raise ValueError(f"The fill_empty_row() function expects the 'time_delta' parameter to be of type <timedelta>, received: {type(time_delta)}")
    if not isinstance(set_index, bool):
        raise ValueError(f"The fill_empty_row() function expects the 'set_index' parameter to be of type <bool>, received: {type(set_index)}")

    nan_columns = reformatted_df.columns[reformatted_df.isna().all()]
    if not nan_columns.empty: print("[WARNING]: fill_empty_rows() -- Columns with all NaN values:", nan_columns.tolist())

    if 'date' not in reformatted_df.columns: reformatted_df.reset_index(inplace=True)

    blank_rows = [] # a list of lists 

    to_filter = reformatted_df[reformatted_df['date'].isna()]
    reformatted_df = reformatted_df[~reformatted_df['date'].isna()]

    for i in range(len(reformatted_df)-1):
        current_timestamp = reformatted_df['date'].iloc[i].replace(second=0, microsecond=0)
        next_timestamp = reformatted_df['date'].iloc[i+1].replace(second=0, microsecond=0)

        while next_timestamp - current_timestamp > time_delta:
            current_timestamp += time_delta
            new_row = _build_empty_row(reformatted_df.columns, current_timestamp)
            if new_row is not None:
                blank_rows.append(new_row)

    if not blank_rows: return reformatted_df      
    else:
        blank_rows_df = pd.DataFrame(blank_rows, columns=reformatted_df.columns)
        reformatted_df = pd.concat([reformatted_df, blank_rows_df]).sort_values('date').reset_index(drop=True)

        reformatted_df.sort_values(by='date', inplace=True)

        pd.concat([to_filter, reformatted_df['date'].isna()])
        reformatted_df = reformatted_df[~reformatted_df['date'].isna()]
        if not to_filter.empty: print("[WARNING]: fill_empty_rows() -- Number of rows identified with invalid (NaT) timestamps:", len(to_filter['date'].tolist()))

        if set_index: reformatted_df.set_index('date', inplace=True)

        return reformatted_df
    