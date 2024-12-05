
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
Accepts a timestamp string and returns a datetime object representing that timestamp.
"""
def get_timestamp(timestamp:str) -> datetime:
    if not isinstance(timestamp, str):
        raise TypeError(f"The 'timestamp' parameter of get_timestamp() should be of type <str>, passed: {type(timestamp)}")

    formatted_timestamp = timestamp[:10] + ' ' + timestamp[11:]
    format_str = "%Y-%m-%d %H:%M:%S"

    return datetime.strptime(formatted_timestamp, format_str) 


"""
Takes the station name and takes optional parameters year and month. Default values for year and month are empty strings.
Returns a string containing the desired file name.
"""
def file_name_builder(station_name, year:str = "", month:str = "") -> str:
    if not isinstance(year, str):
        raise TypeError(f"The 'year' parameter of file_name_builder() should be of type <str>, passed: {type(year)}")
    if not isinstance(month, str):
        raise TypeError(f"The 'month' parameter of file_name_builder() should be of type <str>, passed: {type(month)}")
    
    file_name = station_name[8:14] 
    if year != "" and month != "":
        file_name.append("_" + year + month)
    file_name.append(".dat")

    print(file_name)
    return file_name


"""
Helper function for build_headers() that checks if a header is in the known set of headers.
"""
def headers_are_valid(columns_desired:list, columns_found:list, portal_name:str) -> bool:
    if not isinstance(columns_desired, list):
        raise TypeError(f"The 'columns_desired' parameter of headers_are_valid() should be of type <list>, passed: {type(columns_desired)}")
    if not isinstance(columns_found, list):
        raise TypeError(f"The 'columns_found' parameter of headers_are_valid() should be of type <list>, passed: {type(columns_found)}")
    if not isinstance(portal_name, str):
        raise TypeError(f"The 'portal_name' parameter of headers_are_valid() should be of type <str>, passed: {type(portal_name)}")

    for col in columns_desired:
        if col.endswith('compass_dir'):
            print("\t ======================= ERROR =======================")
            print("\t 'compass_dir' columns are not valid.")
            print(f"\t Only specify shortnames found in {portal_name} associated with the instrument id.")
            return False
        if col not in columns_found:
            print("\t ======================= ERROR =======================")
            print(f"\t Could not locate desired column '{col}' in data stream.")
            str_1 = ""
            for j in columns_found:
                str_1 += f"{j}  "
            print(f"\t Column(s) identified in data stream: {str_1}")
            return False
        
    return True

"""
Helper function for get_columns that applies a kind of sort to the headers. Data from API stream comes into main() via
Python dictionaries, which randomly store key/value pairs, and so the column headers will be randomly ordered without sort.
Takes a list of column names, looks up the sort associated with the portal, and returns a new list of organized header names.
"""
def sort_columns(columns:list) -> list:
    if not isinstance(columns, list):
        raise TypeError(f"The 'columns' parameter in sort_columns() should be of type <list>, passed: {type(columns)}")
   
    sort = [        
        'bt1', 'bp1', 'bh1', 'ht1', 'hh1', 'mt1', 'rg', 'sv1', 'si1', 'su1', 'wd', 'ws' 
    ]

    sorted_columns = [i for i in sort if i in columns] # list of strings
    
    return sorted_columns


"""
Takes a list of dictionaries and returns a list of the set of all variables to be used as columns in the csv.
The exception is when include_test is set to True -- it will append a column 'test' next to each variable indicating 
whether the data is test data. If a wind direction column is in the set, an adjacent compass_dir column will be appended
containing the corresponding compass rose reading for each field.
"""
def get_columns(dictionary_list:list) -> list:
    if not isinstance(dictionary_list, list):
        raise TypeError(f"The 'dictionary_list' parameter in get_columns() should be of type <list>, passed: {type(dictionary_list)}")

    print("\tInside get_columns")
    columns = [] # list of strings 
    for dictionary in dictionary_list:
        cols = list(dictionary.keys())
        print(cols)
        cols_sorted = sort_columns(cols)
        for col in cols_sorted:
            if not col in columns:
                columns.append(str(col))\

    for col in columns:
        print(col)
    return columns

"""
Creates a list of all the header names to be used in the csv requested by user and returns it.
"""
def build_headers(measurements:list) -> list:
    if not isinstance(measurements, list):
        raise TypeError(f"The 'measurements' parameter of build_headers() should be of type <list>, passed: {type(measurements)}")
    
    print(measurements)

    headers = ['year', 'mon', 'day', 'hour', 'min'] # list of strings
    columns = get_columns(measurements)
    print("The columns are:")
    
    if len(columns) == 0: # if no data, pass to main 
        return []
    
    mapper = [ # 'chords shortname', 'reformatted name'
        'year','year',  'mon','mon',  'day','day',  'hour','hour',  'min','min',  
        'htu21d_temp','htu_temp', 'htu21d_humindity','htu_hum',  'bmp_temp','bmp2_temp',  'mcp9808','mcp9808',  'sth31d_temp','sth_temp',  'sth31d_humidity','sth_hum', 
        'bmp_slp','bmp2_slp',  'bmp_pressure','bmp2_pres',  'rain','tipping',  'wind_speed','wind_speed', 'wind_direction', 'wind_dir',  'msl1','bmp2_slp'
        'wg','wg',  'wgd','wgd',  'si1145_vis','vis_light', 'si1145_ir','ir_light', 'si1145_uv','uv_light', 'bpc','bpc'
    ]
    
    for col in columns:
        if col == 'bpc' or col == 'wg' or col == 'wgd':
            continue
        
        headers.append(mapper[mapper.index(col)+1])


    return headers


"""
Accepts an array of headers, timestamps, and of dictionaries containing sensor measurements. 
Also accepts a np array of whether or not  the measurements at that timestamp are test values. 
Accepts a string of the filepath at which to create the csv file and creates the csv there.
Optional parameters "month" and "year" for file gen that isn't daily.
"""
def csv_builder(headers:list, time:np.ndarray, measurements:np.ndarray, filepath:str, null_value, month:int=-1, year:int=-1): 
    if not isinstance(headers, list):
        raise TypeError(f"The 'headers' parameter in csv_builder() should be of type <list>, passed: {type(headers)}")
    if not isinstance(time, np.ndarray):
        raise TypeError(f"The 'time' parameter in csv_builder() should be of type <ndarray>, passed: {type(time)}")
    if not isinstance(measurements, np.ndarray):
        raise TypeError(f"The 'measurements' parameter in csv_builder() should be of type <ndarray>, passed: {type(measurements)}")
    if not isinstance(filepath, str):
        raise TypeError(f"The 'filepath' parameter in csv_builder() should be of type <str>, passed: {type(filepath)}")
    if not isinstance(month, int):
        raise TypeError(f"The 'month' parameter in csv_builder() should be of type <int>, passed: {type(month)}")
    if not isinstance(year, int):
        raise TypeError(f"The 'year' parameter in csv_builder() should be of type <int>, passed: {type(year)}")
    
    mapper = [ # 'chords shortname', 'reformatted name'
        'year','year',  'mon','mon',  'day','day',  'hour','hour',  'min','min',  
        'ht1','htu_temp', 'hh1','htu_hum',  'bt1','bmp2_temp',  'mt1','mcp9808', 
        'bp1','bmp2_pres',  'rg','tipping',  'ws','wind_speed', 'wd', 'wind_dir',  'msl1','bmp2_slp',
        'sv1','vis_light', 'si1','ir_light', 'su1','uv_light',  'sth31d_temp','sth_temp',  'sth31d_humidity','sth_hum',
        'alt','bmp2_alt', 'int','int'
    ]

    if len(time) == len(measurements):
        data = [] # list of timestamps, measurements, and headers (to turn into dataframe)
        for i in range(len(time)):
            og_time = time[i]
            year = og_time[:4] 
            month = og_time[5:7]
            day = og_time[8:10]
            hour = og_time[11:13]
            minute = og_time[14:16]
            measurements[i].update({'year':year, 'mon':month, 'day':day, 'hour':hour, 'min':minute}) 

            keys = list(measurements[i].keys())
            for j in keys:
                if j in mapper: # exclude col's we don't care about (e.g. bpc or css)
                    mapped_key = mapper[mapper.index(j)+1]
                    measurements[i][mapped_key] = measurements[i].pop(j)

            measurement_dict = {header: measurements[i].get(header, null_value) for header in headers} # fill in null value for var's with no data
            data.append(measurement_dict)
        
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(filepath, sep='\t', index=False)
    # else:
    #     raise TimestampError()
    

"""
Accepts np arrays for measurements, time, and test which were created from the data stream from CHORDS, and checks whether 
the transfer of data from the stream to the data structs was a success.
"""
def struct_has_data(measurements:np.ndarray, time:np.ndarray) -> bool:
    if not isinstance(measurements, np.ndarray):
        raise TypeError(f"The 'measurements' parameter in struct_has_data() should be of type <np.ndarray>, passed: {type(measurements)}")
    if not isinstance(time, np.ndarray):
        raise TypeError(f"The 'time' parameter in struct_has_data() should be of type <np.ndarray>, passed: {type(time)}")
    
    flag = True
    if len(measurements) == 0:
        flag = False
    if len(time) == 0:
        flag = False

    return flag


"""
Accepts the resulting data stream of API request, and checks if any of the keys are a known error. Prints are more useful error
message to the screen for troubleshooting. Returns True if error is found, False otherwise.
** when the API returns 'errors' key, the information is stored in a list  **
** when the API returns 'error' key, the information is stored in a string **
"""
def has_errors(all_fields:dict) -> bool:
    if not isinstance(all_fields, dict):
        raise TypeError(f"The 'all_fields' parameter in has_errors() should be of type <dict>, passed: {type(all_fields)}")

    for key in all_fields:
        if key == 'errors' and all_fields['errors'][0] == 'Access Denied, user authentication required.': 
            print(all_fields[key][0])
            print("Check url, email address, and api key.")
            return True
        if key == 'error' and all_fields['error'] == 'Internal Server Error':
            print(all_fields[key])
            print("Check to make sure the instrument ID's are valid. Refer to the CHORDS Portal.")
            return True
                
    return False


"""
Accepts station pressure (sp), altitude (alt), and temperature (temp) to calculate the sea level pressure.
Returns the calculated sea level pressure.
"""
def get_slp(sp:float, alt:float, temp:float) -> float:
    if not isinstance(sp, float):
        raise TypeError(f"The 'sp' parameter in get_slp() should be of type <float>, passed: {type(sp)}")
    if not isinstance(alt, float):
        raise TypeError(f"The 'alt' parameter in get_slp() should be of type <float>, passed: {type(alt)}")
    if not isinstance(temp, float):
        raise TypeError(f"The 'temp' parameter in get_slp() should be of type <float>, passed: {type(temp)}")

    return sp * (1 + ((0.0065*alt) / (temp + (0.0065*alt) + 273.15)))**(-5.257)


def get_compass_dir(degrees):
    if degrees >= 337.5 or degrees < 22.5:
        return 'N'
    elif 22.5 <= degrees < 67.5:
        return 'NE'
    elif 67.5 <= degrees < 112.5:
        return 'E'
    elif 112.5 <= degrees < 157.5:
        return 'SE'
    elif 157.5 <= degrees < 202.5:
        return 'S'
    elif 202.5 <= degrees < 247.5:
        return 'SW'
    elif 247.5 <= degrees < 292.5:
        return 'W'
    elif 292.5 <= degrees < 337.5:
        return 'NW'




"""
Helper function to fill_empty_rows()
The number of columns in each station's csv varies. This function dynamically builds a new row to fill in missing timestamps regardless
of however many columns there might be. Missing rows should be blank.
columns -- the df's column attribute
missing_timestamp -- the timestamp for the missing row
"""
def _build_empty_row(columns:Index, missing_timestamp:Timestamp) -> list:
    if not isinstance(columns, Index):
        raise TypeError(f"The build_empty_row() function expects 'columns' parameter to be of type <Index>, passed: {type(columns)}")
    if not isinstance(missing_timestamp, Timestamp):
        raise TypeError(f"The build_empty_row() function expects 'missing_timestamp' parameter to be of type <Timestamp>, passed: {type(missing_timestamp)}")
    
    row_builder_lookup = [ # all the possible column names
        'bmp2_temp',np.nan,  'bmp2_pres',np.nan,  'bmp2_slp',np.nan,  'bme2_hum',np.nan,  'htu_temp',np.nan,  'htu_hum',np.nan,  
        'mcp9808',np.nan,  'tipping',np.nan,  'wind_dir',np.nan,  'wind_speed',np.nan,  # 3D-PAWS

        'year_month',np.nan,  'year_month_day',np.nan,  'year_month_day_hour',np.nan,

        'temp',np.nan,  'humidity',np.nan,  'actual_pressure',np.nan,  'sea_level_pressure',np.nan,  'avg_wind_dir',np.nan,
        'avg_wind_speed',np.nan,  'total_rainfall',np.nan  # TSMS
    ] # using lookup table, even tho it's redundant, because it can check for missing columns

    next_row = [missing_timestamp]
    for col in columns:
        if col == 'date':
            continue
        try:
            index = row_builder_lookup.index(col)
            next_row.append(row_builder_lookup[index+1])
        except ValueError:
            print(f"Could not find {col} in lookup table.")
            sys.exit(1)

    return next_row


"""
Create a list to merge with the reformatted dataframe that fills in data gaps with missing timestamps. 
Merges that list with the reformatted dataframe and returns it. 
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

    if 'date' not in reformatted_df.columns: reformatted_df.reset_index(inplace=True)

    blank_rows = [] # a list of lists

    for i in range(len(reformatted_df)-1):
        current_timestamp = reformatted_df['date'].iloc[i]
        next_timestamp = reformatted_df['date'].iloc[i+1]

        while next_timestamp - current_timestamp > time_delta:
            current_timestamp += time_delta
            new_row = _build_empty_row(reformatted_df.columns, current_timestamp)
            if new_row is not None:
                blank_rows.append(new_row)
            
    if blank_rows:
        blank_rows_df = pd.DataFrame(blank_rows, columns=reformatted_df.columns)
        reformatted_df = pd.concat([reformatted_df, blank_rows_df]).sort_values('date').reset_index(drop=True)

    if set_index: reformatted_df.set_index('date', inplace=True)

    return reformatted_df
