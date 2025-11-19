"""
Data filter. Creates final csv's as well as outlier csv's. Contains plot gen logic.

Requires station_TSMS00 -> 08 folder struct, with the complete records for TSMS and 3D-PAWS within each folder
within the data_origin directory. Alongside the station_TSMS00 -> 08 folders, have a folder for each site.
Note:   The subfolders for station_TSMS00 -> 08 are generated automatically.
        The subfolders for sites Ankara, Konya, and Adana are NOT generated automatically.

The data_destination pathway doesn't require this, just list the full pathname where you want things stored.

Phase 1: Nulls (-999.99)
Phase 2: Timestamp resets (out of order timestamps)
Phase 3: Thresholds (unrealistic values)
Phase 4: Manual removal of identified special cases
Phase 5: Statistical outlier removal (rolling z-score and Hampel filter)

How to use this script:
    The outlier removal logic in the first portion of this script is to remain uncommented.
    The plot generating scripts which follow the outlier removal portion should have only the 1 logic block uncommented.
    Each section of plot gen logic is separated by a header. The following is an example:
        =============================================================================================================================
        Create a time series plot of the 3D PAWS station data versus the TSMS reference station. MONTHLY RECORDS
        =============================================================================================================================
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from windrose import WindroseAxes
import matplotlib.cm as cm
import numpy as np
from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_squared_error
import seaborn as sns
from pathlib import Path
from datetime import timedelta, datetime
from functools import reduce

sys.path.append(str(Path(__file__).resolve().parents[2]))
from resources import functions as func


data_origin = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/reformatted/CSV_Format/analysis/"
data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/stats-plots/finale/"

outlier_reasons = [
    "null", "timestamp_reset", "threshold", "manual_removal", "htu_trend_switch", "z-score_contextual",
    "hampel_contextual"
]

station_variables = [ 
    "temperature", "humidity", "actual_pressure", "sea_level_pressure", "wind", "total_rainfall"
]
variable_mapper = { # TSMS : 3DPAWS
    "temperature":["bmp2_temp", "htu_temp", "mcp9808"],
    "humidity":["bme2_hum", "htu_hum"],
    "actual_pressure":["bmp2_pres"],
    "sea_level_pressure":["bmp2_slp"],
    "avg_wind_dir":["wind_dir"],
    "avg_wind_speed":["wind_speed"], 
    "total_rainfall":["tipping"]
}
station_directories = [ # i really oughta do pathlib for os agnostic code, this is silly
    "station_TSMS00/", "station_TSMS01/", "station_TSMS02/",   
    "station_TSMS03/", "station_TSMS04/", "station_TSMS05/",
    "station_TSMS06/", "station_TSMS07/", "station_TSMS08/"
]
# station_directories = [
#     "station_TSMS00\\", "station_TSMS01\\", "station_TSMS02\\",   
#     "station_TSMS03\\", "station_TSMS04\\", "station_TSMS05\\",
#     "station_TSMS06\\", "station_TSMS07\\", "station_TSMS08\\"
# ]

station_files = [] # list of lists of filenames under each station directory [[TSMS00 files], [TSMS01 files], [TSMS02 files]...]
for station in station_directories:
    station_files.append(sorted([f for f in os.listdir(data_origin + station) # sorted by filename
                                    if os.path.isfile(os.path.join(data_origin + station, f)) and f != ".DS_Store"]))
    
# set up folder struct for each station where data is outputted
for i in range(len(station_directories)):
    try:
        os.makedirs(data_destination+station_directories[i], exist_ok=True)
        for file in station_files[i]:
            try:
                for var in station_variables:
                    os.makedirs(data_destination+station_directories[i]+var+'/', exist_ok=True)
            except Exception as e:
                print("Error creating the directory: ", var)
    except Exception as e:
        print("Could not create ", station_directories[i][:len(station_directories[i])-1])
    
# -------------------------------------------------------------------------------------------------------------------------------
paws_dfs = []
tsms_dfs = []

for i in range(len(station_directories)):
    print("\n----------------------")
    paws_df = tsms_df = None
    """
    =============================================================================================================================
    Create the 3D PAWS and TSMS dataframes.
    =============================================================================================================================
    """
    for file in station_files[i]:
        if file.startswith("TSMS"): # 3D-PAWS records start w/ 'TSMS', TSMS ref. records start w/ the site name
            paws_df = pd.read_csv( 
                            data_origin+station_directories[i]+file,
                            header=0,
                            low_memory=False
                        )
            paws_df.columns = paws_df.columns.str.strip()

            paws_df['date'] = pd.to_datetime(paws_df['date'])
            
            paws_df.drop( # no reference data for these col's OR no data exists in these columns (e.g. sth) OR scalar (e.g. alt)
                        ['int', 'sth_temp', 'sth_hum', 'bmp2_alt'], 
                        axis=1, 
                        inplace=True
                    )
            
            paws_df = paws_df[['date', 'bmp2_temp', 'htu_temp', 'mcp9808', 'bme2_hum', 'htu_hum', 'bmp2_pres',
                                'bmp2_slp', 'wind_dir', 'wind_speed', "tipping"]]
            
            paws_df['year_month'] = paws_df['date'].dt.to_period('M')
            paws_df['year_month_day'] = paws_df['date'].dt.to_period('D')
            paws_df['year_month_day_hour'] = paws_df['date'].dt.to_period('h')
            
            paws_df.set_index('date', inplace=True)

        else:
            tsms_df = pd.read_csv(
                            data_origin+station_directories[i]+file,
                            header=0,
                            low_memory=False
                        )
            tsms_df.columns = tsms_df.columns.str.strip()

            tsms_df['date'] = pd.to_datetime(tsms_df['date'])

            tsms_df = tsms_df[['date', 'temperature', 'humidity', 'actual_pressure', 'sea_level_pressure', 
                                'avg_wind_dir', 'avg_wind_speed', 'total_rainfall']]
            
            tsms_df['year_month'] = tsms_df['date'].dt.to_period('M')
            tsms_df['year_month_day'] = tsms_df['date'].dt.to_period('D')
            tsms_df['year_month_day_hour'] = tsms_df['date'].dt.to_period('h')

            tsms_df.set_index('date', inplace=True) 

    
    warnings.filterwarnings( # Ignore warnings so they don't clog up the console output
        "ignore", 
        category=FutureWarning, 
        message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated."
    )

    
    """
    =============================================================================================================================
    =============================================================================================================================
    =============================================================================================================================
                Eliminate outliers from the dataset and store in separate dataframe for each station.
            Creates a FILTERED dataframe, where all cells containing an outlier have been replaced with np.nan
    =============================================================================================================================
    =============================================================================================================================
    =============================================================================================================================
    """
    
    paws_outliers = pd.DataFrame(columns=['date', 'column_name', 'original_value', 'outlier_type'])
    tsms_outliers = pd.DataFrame(columns=['date', 'column_name', 'original_value', 'outlier_type'])
    
    paws_df.reset_index(inplace=True)
    tsms_df.reset_index(inplace=True)

    paws_df_FILTERED = paws_df.copy()
    tsms_df_FILTERED = tsms_df.copy()

    paws_df.set_index('date', inplace=True)
    tsms_df.set_index('date', inplace=True)

    print()
    print("Starting outlier removal for ", station_directories[i][8:14])
    

    """
    =============================================================================================================================
    Phase 1: Filter out nulls (-999.99)
    =============================================================================================================================
    """
    print("Phase 1: Filtering out nulls (-999.99).")

    for variable in variable_mapper:                        # Filtering TSMS ----------------------------------------------------
        mask_999 = tsms_df_FILTERED[variable] == -999.99    # Replace -999.99 with np.nan
        tsms_df_FILTERED.loc[mask_999, variable] = np.nan
        
        mask_null = tsms_df_FILTERED[variable].isnull()     # Identify nulls
        
        if mask_null.any():                                 # Add nulls to outliers_df
            outliers_to_add = pd.DataFrame({
                'date': tsms_df_FILTERED['date'][mask_null],
                'column_name': variable,
                'original_value': -999.99,
                'outlier_type': outlier_reasons[0]
            })
            tsms_outliers = pd.concat([tsms_outliers, outliers_to_add], ignore_index=True)
        

        outliers_to_add = None
        

        for var in variable_mapper[variable]:               # Filtering PAWS ----------------------------------------------------
            mask_999 = paws_df_FILTERED[var] == -999.99   
            paws_df_FILTERED.loc[mask_999, var] = np.nan

            mask_null = paws_df_FILTERED[var].isnull()    

            if mask_null.any():                  
                outliers_to_add = pd.DataFrame({
                    'date': paws_df_FILTERED['date'][mask_null],
                    'column_name': var,
                    'original_value': -999.99,
                    'outlier_type': outlier_reasons[0]
                })
                paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)
    

    """
    =============================================================================================================================
    Phase 2: Filter out time resets
    =============================================================================================================================
    """
    print(f"Phase 2: Filtering out time resets.")

    tsms_df_FILTERED = tsms_df_FILTERED.sort_values(by='date')  # Filtering TSMS ------------------------------------------------
    tsms_df_FILTERED['time_diff'] = tsms_df_FILTERED['date'].diff()
    
    out_of_order_tsms = tsms_df_FILTERED[tsms_df_FILTERED['time_diff'] <= pd.Timedelta(0)]  # Identify out-of-order timestamps

    tsms_outliers = pd.concat([tsms_outliers, pd.DataFrame({    # Add out-of-order timestamps to outlier dataframe
        'date': out_of_order_tsms['date'],
        'column_name': 'date',
        'original_value': np.nan,
        'outlier_type': outlier_reasons[1]
    })], ignore_index=True)

    df_sequential = tsms_df_FILTERED[tsms_df_FILTERED['time_diff'] > pd.Timedelta(0)]       # Filter out-of-order timestamps
    tsms_df_FILTERED = df_sequential.drop(columns=['time_diff'])

    df_sequential = None  

    paws_df_FILTERED = paws_df_FILTERED.sort_values(by='date')  # Filtering PAWS ------------------------------------------------
    paws_df_FILTERED['time_diff'] = paws_df_FILTERED['date'].diff()
    
    out_of_order_paws = paws_df_FILTERED[paws_df_FILTERED['time_diff'] <= pd.Timedelta(0)]  

    paws_outliers = pd.concat([paws_outliers, pd.DataFrame({    
        'date': out_of_order_paws['date'],
        'column_name': 'date',
        'original_value': np.nan,
        'outlier_type': outlier_reasons[1]
    })], ignore_index=True)

    df_sequential = paws_df_FILTERED[paws_df_FILTERED['time_diff'] > pd.Timedelta(0)]       
    paws_df_FILTERED = df_sequential.drop(columns=['time_diff'])


    """
    =============================================================================================================================
    Phase 3: Filter out unrealistic values
    =============================================================================================================================
    """
    print(f"Phase 3: Filtering out unrealistic values.") 

    for variable in variable_mapper:                                # Filtering TSMS --------------------------------------------
        existing_nulls = tsms_df_FILTERED[variable].isnull()        # Create a mask for existing nulls before applying the filter
        
        if variable == "temperature":
            mask_temp = (tsms_df_FILTERED[variable] > 50) | (tsms_df_FILTERED[variable] < -50)
            tsms_df_FILTERED.loc[mask_temp, variable] = np.nan
        elif variable == "humidity":
            mask_hum = (tsms_df_FILTERED[variable] > 100) | (tsms_df_FILTERED[variable] < 0)
            tsms_df_FILTERED.loc[mask_hum, variable] = np.nan
        elif variable == "actual_pressure" or variable == "sea_level_pressure":
            mask_pres = (tsms_df_FILTERED[variable] > 1084) | (tsms_df_FILTERED[variable] < 870)
            tsms_df_FILTERED.loc[mask_pres, variable] = np.nan
        elif variable == "avg_wind_speed":
            mask_wndspd = (tsms_df_FILTERED[variable] > 100) | (tsms_df_FILTERED[variable] < 0)
            tsms_df_FILTERED.loc[mask_wndspd, variable] = np.nan
        elif variable == "avg_wind_dir":
            mask_wnddir = (tsms_df_FILTERED[variable] > 360) | (tsms_df_FILTERED[variable] < 0)
            tsms_df_FILTERED.loc[mask_wnddir, variable] = np.nan
        elif variable == "total_rainfall":
            mask_rain = (tsms_df_FILTERED[variable] > 32) | (tsms_df_FILTERED[variable] < 0)
            tsms_df_FILTERED.loc[mask_rain, variable] = np.nan
        else:
            print(f"[ERROR]: {variable} not found in dataframe.")
            sys.exit(-1)
        
        new_nulls = tsms_df_FILTERED[variable].isnull() & ~existing_nulls   # Create a mask for new nulls after applying the filter
        
        outliers_to_add = pd.DataFrame({                                    # Add new nulls to the outlier dataframe
            'date': tsms_df_FILTERED.loc[new_nulls, 'date'],
            'column_name': variable,
            'original_value': tsms_df_FILTERED.loc[new_nulls, variable],
            'outlier_type': outlier_reasons[2]
        })
        tsms_outliers = pd.concat([tsms_outliers, outliers_to_add], ignore_index=True)
        
        outliers_to_add = None
        new_nulls = None

        for var in variable_mapper[variable]:               # Filtering PAWS ----------------------------------------------------
            existing_nulls = paws_df_FILTERED[var].isnull()

            if var == "bmp2_temp" or var == "htu_temp" or var == "mcp9808":
                mask_temp = (paws_df_FILTERED[var] > 50) | (paws_df_FILTERED[var] < -50)
                paws_df_FILTERED.loc[mask_temp, var] = np.nan
            elif var == "htu_hum" or var == "bme2_hum":
                mask_hum = (paws_df_FILTERED[var] > 100) | (paws_df_FILTERED[var] < 0)
                paws_df_FILTERED.loc[mask_hum, var] = np.nan
            elif var == "bmp2_pres" or var == "bmp2_slp":
                mask_pres = (paws_df_FILTERED[var] > 1084) | (paws_df_FILTERED[var] < 870)
                paws_df_FILTERED.loc[mask_pres, var] = np.nan
            elif var == "wind_speed":
                paws_df_FILTERED[var] = pd.to_numeric(paws_df_FILTERED[var], errors='coerce')
                mask_wndspd = (paws_df_FILTERED[var] > 100) | (paws_df_FILTERED[var] < 0)
                paws_df_FILTERED.loc[mask_wndspd, var] = np.nan
            elif var == "wind_dir":
                mask_wnddir = (paws_df_FILTERED[var] > 360) | (paws_df_FILTERED[var] < 0)
                paws_df_FILTERED.loc[mask_wnddir, var] = np.nan
            elif var == "tipping":
                mask_rain = (paws_df_FILTERED[var] > 32) | (paws_df_FILTERED[var] < 0)
                paws_df_FILTERED.loc[mask_rain, var] = np.nan
            else:
                print("Error with ", var)
            
            new_nulls = paws_df_FILTERED[var].isnull() & ~existing_nulls  
            
            outliers_to_add = pd.DataFrame({               
                'date': paws_df_FILTERED.loc[new_nulls, 'date'],
                'column_name': var,
                'original_value': paws_df_FILTERED.loc[new_nulls, var],
                'outlier_type': outlier_reasons[2]
            })
            paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)


    import operator as o
    """
    ============================================================================================================================
    Phase 4: Manual removal of known outliers.
    ============================================================================================================================
    """
    print(f"Phase 4: Manual removal of known outliers (see Phase 4 in code for notes).") 

    """
    NOTE: The outliers specified in the station_rules variable fall exclusively under these categories:
            - test data generated during the fabrication process
            - erroneous spikes (e.g. random false zero's) which weren't removed during the statistical 
                outlier cleaning sweep

        Below station_rules, there are 2 other removals of erroneous rainfall data. The precipitation 
        data removed was determined to be the result of a faulty connection (TSMS08) and test data 
        generated during the fabrication process (TSMS04).

        IMPORTANT:
        Not removed from the final dataset includes erroneous HTU temperature and r. humidity data.
        The HTU was observed in stations TSMS00 -> TSMS04 to suffer from a "bit-switching" issue, 
        where the temperature trend would toggle between valid temperature readings and erroneous 
        r. humidity readings at 1 -> 4 minute intervals. The same is true of the r. humidity trend --
        valid r. humidity readings would be interspersed with erroneous temperature data. 
        Timescales on the order of a few hours up to 2 months would be continuously affected by this 
        sensor malfunction. Unfortunately due to time constraints, this data could not be removed. 

        A signal processing approach is proposed for future cleaning attempts. Complications arise 
        given the nature of the rate at which the "bit-switching" noise occurs -- because the rate 
        of noise is similar to the sampling rate of the weather stations, standard signal processing 
        techniques will run the risk of causing aliasing.
    """

    df = paws_df_FILTERED

    station_rules = {
        "TSMS00" : { 
            'bmp2_temp': [([pd.Period('2023-02')], o.gt, 16)],
            'htu_temp': [([pd.Period('2023-02')], o.gt, 16)],
            'mcp9808': [([pd.Period('2023-02')], o.gt, 16)],
            'bmp2_pres': [([pd.Period('2022-08')], o.gt, 1050)],
            'bmp2_slp': [([pd.Period('2022-08'), pd.Period('2022-09')], o.lt, 980)]
        },
        "TSMS01": {},
        "TSMS02": { 
            'bmp2_temp': [([pd.Period('2024-02')], o.gt, 20)],
            'bmp2_pres': [([pd.Period('2024-02')], o.gt, 930)],
            'bmp2_slp': [([pd.Period('2024-02')], o.lt, 1015)]
        },
        "TSMS03": { 
            'htu_temp': [([pd.Period('2022-10')], o.lt, -30)],
            'bmp2_slp': [([pd.Period('2024-06')], o.lt, 980)]
        },
        "TSMS04": {},
        "TSMS05": {
            'bmp2_temp': [([pd.Period('2024-01')], o.gt, 17)],
            'mcp9808': [([pd.Period('2024-01')], o.gt, 17)]
        },
        "TSMS06": {
            'bmp2_temp': [
                ([pd.Period('2024-01')], o.gt, 25),
                ([pd.Period('2024-01')], o.lt, 1),
                ([pd.Period('2024-09'), pd.Period('2024-10')], o.lt, 5)
            ],
            'htu_temp': [([pd.Period('2024-09'), pd.Period('2024-10')], o.lt, 5)],
            'mcp9808': [([pd.Period('2024-09'), pd.Period('2024-10')], o.lt, 5)],
            'htu_hum': [([pd.Period('2024-09'), pd.Period('2024-10')], o.lt, 5)]
        },
        "TSMS07": {
            'bmp2_temp': [
                ([pd.Period('2022-10')], lambda s, _: s.notna(), None),
                ([pd.Period('2024-09'), pd.Period('2024-10')], o.lt, 5)
            ],
            'htu_temp': [([pd.Period('2022-10')], lambda s, _: s.notna(), None)],
            'mcp9808': [([pd.Period('2022-10')], lambda s, _: s.notna(), None)],
            'htu_hum': [([pd.Period('2022-10')], lambda s, _: s.notna(), None)],
            'bmp2_pres': [
                ([pd.Period('2022-10')], lambda s, _: s.notna(), None),
                ([pd.Period('2022-11')], o.lt, 1000)
            ],
            'bmp2_slp': [
                ([pd.Period('2022-10')], lambda s, _: s.notna(), None),
                ([pd.Period('2022-11')], o.lt, 1000)
            ]
        },
        "TSMS08": {
            'bmp2_temp': [
                (None, lambda df: df['date'].between('2022-10-01','2022-11-05'), None),
                ([pd.Period('2024-10')], o.lt, 5)
            ],
            'htu_temp': [
                (None, lambda df: df['date'].between('2022-10-01','2022-11-05'), None),
                ([pd.Period('2024-01')], o.lt, 1)
            ],
            'mcp9808': [
                (None, lambda df: df['date'].between('2022-10-01','2022-11-05'), None)
            ],
            'htu_hum': [
                (None, lambda df: df['date'].between('2022-10-01','2022-11-05'), None),
                ([pd.Period('2024-01')], o.lt, 1)
            ],
            'bmp2_pres': [
                (None, lambda df: df['date'].between('2022-10-01','2022-11-05'), None),
                ([pd.Period('2022-11')], o.lt, 980)
            ],
            'bmp2_slp': [
                (None, lambda df: df['date'].between('2022-10-01','2022-11-05'), None),
                ([pd.Period('2022-11')], o.lt, 980)
            ]
        }
    }

    station = station_directories[i][8:14]
    cleanup = station_rules.get(station)

    for sensor, rules in cleanup.items():
        df[sensor] = pd.to_numeric(df[sensor], errors="coerce")
        existing_nulls = df[sensor].isna()
        original = df[sensor].copy()

        for months, cmpfunc, thresh in rules:
            if months is None:
                mask = cmpfunc(df)
            else:
                mask_period = df["year_month"].isin(months)

                if thresh is None:
                    mask_value = original.notna()
                else:
                    mask_value = cmpfunc(original, thresh)

                mask = mask_period & mask_value

            new_null = mask & ~existing_nulls
            if new_null.any():
                to_log = original.loc[new_null]
                paws_outliers = pd.concat([
                    paws_outliers,
                    pd.DataFrame({
                        "date":           df.loc[new_null, "date"],
                        "column_name":    sensor,
                        "original_value": to_log,
                        "outlier_type":   outlier_reasons[3],
                    })
                ], ignore_index=True)

                df.loc[mask, sensor] = np.nan
                existing_nulls |= mask

    # Erroneously high values -- most likely test tips during fabrication process
    if station_directories[i][8:14] == "TSMS04":
        existing_nulls = paws_df_FILTERED['tipping'].isnull() 

        exclude_rainfall_start = pd.to_datetime(paws_df_FILTERED['date'].iloc[0])
        exclude_rainfall_end = pd.to_datetime("2022-08-31 23:59:59")

        to_remove = paws_df_FILTERED.loc[
            (paws_df_FILTERED['date'] >= exclude_rainfall_start) & (paws_df_FILTERED['date'] <= exclude_rainfall_end), 'tipping'
        ].copy()

        paws_df_FILTERED.loc[
            (paws_df_FILTERED['date'] >= exclude_rainfall_start) & (paws_df_FILTERED['date'] <= exclude_rainfall_end), 'tipping'
        ] = np.nan

        new_nulls = paws_df_FILTERED['tipping'].isnull() & ~existing_nulls

        outliers_to_add = pd.DataFrame({                                
            'date': paws_df_FILTERED.loc[new_nulls, 'date'],
            'column_name': 'tipping',
            'original_value': to_remove,
            'outlier_type': outlier_reasons[3]
        })
        paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)

    # Erroneously high values -- possibly faulty connector, or manual manipulation of the gauge by the public
    if station_directories[i][8:14] == "TSMS08":
        paws_df_FILTERED['tipping'] = pd.to_numeric(paws_df_FILTERED['tipping'], errors='coerce')
        existing_nulls = paws_df_FILTERED['tipping'].isnull() 
        original = paws_df_FILTERED['tipping'].copy()

        dates_to_remove = [
            ("2023-04-08","2023-04-14 23:59:00"),        
            ("2023-05-15 00:00:00","2023-05-17 23:59:59"), 
            ("2024-02-08 00:00:00","2024-02-11 23:59:59"), 
            ("2024-02-15 00:00:00","2024-02-20 23:59:59"), 
            ("2024-03-07 00:00:00","2024-03-07 23:59:59"),
            ("2024-03-09 00:00:00","2024-03-10 23:59:59"), 
            ("2024-03-21 00:00:00","2024-03-21 23:59:59"),
        ]

        to_remove_list = []
        for start_date, end_date in dates_to_remove:
            exclude_rainfall_start = pd.to_datetime(start_date)
            exclude_rainfall_end = pd.to_datetime(end_date)

            mask = (paws_df_FILTERED['date'] >= exclude_rainfall_start) & (paws_df_FILTERED['date'] <= exclude_rainfall_end)
            to_remove = paws_df_FILTERED.loc[mask, 'tipping'].copy()
            to_remove_list.append(to_remove)

            paws_df_FILTERED.loc[
                (paws_df_FILTERED['date'] >= exclude_rainfall_start) & (paws_df_FILTERED['date'] <= exclude_rainfall_end), 'tipping'
            ] = np.nan

        new_nulls = paws_df_FILTERED['tipping'].isnull() & ~existing_nulls

        all_removed = pd.concat(to_remove_list)
        all_removed = all_removed[new_nulls]

        outliers_to_add = pd.DataFrame({                                
            'date': paws_df_FILTERED.loc[new_nulls, 'date'],
            'column_name': 'tipping',
            'original_value': original.loc[new_nulls],
            'outlier_type': outlier_reasons[3]
        })
        paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)

    
    """
    =============================================================================================================================
    Phase 5: Filter out contextual outliers using statistical methods.
    =============================================================================================================================
    """
    print(f"Phase 5: Filtering out contextual outliers using rolling Z-score and Hampel filter.")

    tsms_df_FILTERED.reset_index(drop=True, inplace=True)         
    paws_df_FILTERED.reset_index(drop=True, inplace=True)

    exclude_htu_noise = {   # Exclude HTU bit-switching from analysis for temperature & rh; 3D-PAWS only    
        "TSMS00": [ 
            "2022-08", "2022-12", "2023-01", "2023-06", "2023-07", "2023-08", "2023-09", 
            "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04", 
            "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11"
        ],
        "TSMS01": [
            "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04", "2023-05",
            "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2024-01",
            "2024-03", "2024-04", "2024-05", "2024-11"
        ],
        "TSMS02": [
            "2022-09", "2022-11", "2022-12", "2023-01", "2023-04", "2023-05", "2023-06",
            "2023-08", "2023-09", "2023-10", "2023-11"
        ],
        "TSMS03": [
            "2023-02", "2023-03", "2023-04"
        ],
        "TSMS04": [
            "2024-11"
        ]
    }

    for variable in variable_mapper:                                # Filtering TSMS --------------------------------------------
        if variable in ["avg_wind_speed", "avg_wind_dir", "total_rainfall"]: continue

        tsms_df_FILTERED[variable] = pd.to_numeric(tsms_df_FILTERED[variable], errors='coerce')
        existing_nulls = tsms_df_FILTERED[variable].isna()

        original = tsms_df_FILTERED[variable].copy()
        stats_base = original.copy()

        window = 20  
        threshold = 3

        rolling_mean = stats_base.rolling(window=window, center=True, min_periods=window//2).mean() 
        rolling_std = stats_base.rolling(window=window, center=True, min_periods=window//2).std()
        rolling_med = stats_base.rolling(window=window, center=True, min_periods=window//2).median()

        rolling_std = rolling_std.replace(0, np.nan) # don't divide by zero

        z_score = (original - rolling_mean) / rolling_std
        mad = (original - rolling_med).abs().rolling(window=window, center=True, min_periods=window//2).median()

        hampel_threshold = 3 * 1.4826 * mad # 1.4826 rescales median absolute deviation 

        mask_z = z_score.abs() > threshold
        mask_h = (original - rolling_med).abs() > hampel_threshold
        new_null_z = mask_z & ~existing_nulls
        new_null_h = mask_h & ~existing_nulls
        
        outliers_to_add = pd.DataFrame({
            'date': tsms_df_FILTERED.loc[new_null_z, 'date'],
            'column_name': variable,
            'original_value': original.loc[new_null_z],
            'outlier_type': outlier_reasons[5]
        })
        tsms_outliers = pd.concat([tsms_outliers, outliers_to_add], ignore_index=True)
        outliers_to_add = pd.DataFrame({
            'date': tsms_df_FILTERED.loc[new_null_h, 'date'],
            'column_name': variable,
            'original_value': original.loc[new_null_h],
            'outlier_type': outlier_reasons[6]
        })
        tsms_outliers = pd.concat([tsms_outliers, outliers_to_add], ignore_index=True)

        tsms_df_FILTERED.loc[new_null_z, variable] = np.nan
        tsms_df_FILTERED.loc[new_null_h, variable] = np.nan

        outliers_to_add = None

        for var in variable_mapper[variable]:               # Filtering PAWS ----------------------------------------------------
            paws_df_FILTERED[var] = pd.to_numeric(paws_df_FILTERED[var], errors='coerce')
            existing_nulls = paws_df_FILTERED[var].isnull()

            original = paws_df_FILTERED[var].copy()

            # exclude certain timeframes for HTU due to noise contamination
            mask_exclude = pd.Series(False, index=paws_df_FILTERED.index)
            if var in ['htu_temp', 'htu_hum']:
                bad_months = exclude_htu_noise.get(station_directories[i][8:14], [])
                mask_exclude = paws_df_FILTERED['year_month'].isin(bad_months)
            
            stats_base = original.copy()
            stats_base.loc[mask_exclude] = np.nan

            rolling_mean = stats_base.rolling(window=window, center=True, min_periods=window//2).mean()
            rolling_std = stats_base.rolling(window=window, center=True, min_periods=window//2).std()
            rolling_med = stats_base.rolling(window=window, center=True, min_periods=window//2).median()

            rolling_std = rolling_std.replace(0, np.nan)

            z_score = (original - rolling_mean) / rolling_std
            mad = (original - rolling_med).abs().rolling(window=window, center=True, min_periods=window//2).median()

            hampel_threshold = 3 * 1.4826 * mad

            mask_z = z_score.abs() > threshold
            mask_h = (original - rolling_med).abs() > hampel_threshold
            new_null_z = mask_z & ~existing_nulls & ~mask_exclude
            new_null_h = mask_h & ~existing_nulls & ~mask_exclude

            outliers_to_add = pd.DataFrame({
                'date': paws_df_FILTERED.loc[new_null_z, 'date'],
                'column_name': var,
                'original_value': original.loc[new_null_z],
                'outlier_type': outlier_reasons[5]
            })
            paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)
            outliers_to_add = pd.DataFrame({
                'date': paws_df_FILTERED.loc[new_null_h, 'date'],
                'column_name': var,
                'original_value': original.loc[new_null_h],
                'outlier_type': outlier_reasons[6]
            })
            paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)

            paws_df_FILTERED.loc[new_null_z, var] = np.nan
            paws_df_FILTERED.loc[new_null_h, var] = np.nan

    paws_dfs.append(paws_df_FILTERED)
    tsms_dfs.append(tsms_df_FILTERED)

    """
    =========================================================================================================
    CREATE FINAL CSV'S
    =========================================================================================================
    """
    # paws_df_FILTERED.drop(columns=['bme2_hum','year_month', 'year_month_day', 'year_month_day_hour'], inplace=True)

    # if i in [0,1,2]: tsms_outliers.to_csv(data_destination+f"TSMS-Reference_Ankara_outliers.csv", index=False)
    # elif i in [3,4,5]: tsms_outliers.to_csv(data_destination+f"TSMS-Reference_Konya_outliers.csv", index=False)
    # elif i in [6,7,8]: tsms_outliers.to_csv(data_destination+f"TSMS-Reference_Adana_outliers.csv", index=False)
    # paws_outliers.to_csv(data_destination+f"3DPAWS_{station_directories[i][8:14]}_outliers.csv", index=False)

    # if i in [0,1,2]: tsms_df_FILTERED.to_csv(data_destination+f"TSMS-Reference_Ankara_final.csv", index=False)
    # elif i in [3,4,5]: tsms_df_FILTERED.to_csv(data_destination+f"TSMS-Reference_Konya_final.csv", index=False)
    # elif i in [6,7,8]: tsms_df_FILTERED.to_csv(data_destination+f"TSMS-Reference_Adana_final.csv", index=False)
    # paws_df_FILTERED.to_csv(data_destination+f"3DPAWS_{station_directories[i][8:14]}_final.csv", index=False)

    print("\n----------------------")

# ----------------------------------------------------------------------------------------------------------------------------------------------- PLOT GEN LOGIC
    
    """
    =============================================================================================================================
    Create a time series plot of the 3D PAWS station data versus the TSMS reference station. MONTHLY RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: " \
    #         "Time-series plots for temperature, humidity, actual pressure and sea level pressure -- monthly records")
    
    # paws_df_FILTERED.set_index('date', inplace=True)
    # tsms_df_FILTERED.set_index('date', inplace=True)
    
    # tsms_df_FILTERED = tsms_df_FILTERED[paws_df_FILTERED.index[0]:paws_df_FILTERED.index[-1]]

    # # Temperature ---------------------------------------------------------------------------------------------------------------
    # print("Temperature per sensor")
    # for t in variable_mapper['temperature']:
    #     for year_month, paws_group in paws_df_FILTERED.groupby('year_month'):
    #         tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

    #         plt.figure(figsize=(12, 6))

    #         plt.plot(paws_group.index, paws_group[f'{t}'], marker='.', markersize=1, label=f"3D PAWS {t}")
    #         plt.plot(tsms_group.index, tsms_group['temperature'], marker='.', markersize=1, label='TSMS')
        
    #         plt.title(f'{station_directories[i][8:14]} Temperature for {year_month}: 3D PAWS {t} versus TSMS')
    #         plt.xlabel('Date')
    #         plt.ylabel('Temperature (˚C)')
    #         plt.xticks(rotation=45)

    #         plt.legend()

    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.savefig(data_destination+station_directories[i]+f"temperature/{station_directories[i][8:14]}_{t}_{year_month}.png")
    #         #plt.savefig(data_destination+station_directories[i]+f"temperature\\{station_directories[i][8:14]}_{t}_{year_month}.png")
            
    #         plt.clf()
    #         plt.close()


    # # Humidity ------------------------------------------------------------------------------------------------------------------
    # print("Humidity per sensor")
    # for h in variable_mapper['humidity']:
    #     if h == "bme2_hum": continue
    #     for year_month, paws_group in paws_df_FILTERED.groupby('year_month'):
    #         tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

    #         plt.figure(figsize=(12, 6))

    #         plt.plot(paws_group.index, paws_group[f'{h}'], marker='.', markersize=1, label=f"3D PAWS {h}")
    #         plt.plot(tsms_group.index, tsms_group['humidity'], marker='.', markersize=1, label='TSMS')
        
    #         plt.title(f'{station_directories[i][8:14]} Humidity for {year_month}: 3D PAWS {h} versus TSMS')
    #         plt.xlabel('Date')
    #         plt.ylabel('Humidity (%)')
    #         plt.xticks(rotation=45)

    #         plt.legend()

    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.savefig(data_destination+station_directories[i]+f"humidity/{station_directories[i][8:14]}_{h}_{year_month}.png")
    #         #plt.savefig(data_destination+station_directories[i]+f"humidity\\{station_directories[i][8:14]}_{h}_{year_month}.png")
            
    #         plt.clf()
    #         plt.close()


    # # Pressure ------------------------------------------------------------------------------------------------------------------
    # print("Pressure per sensor")
    # for year_month, paws_group in paws_df_FILTERED.groupby("year_month"):
    #     tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

    #     plt.figure(figsize=(12, 6))

    #     plt.plot(paws_group.index, paws_group['bmp2_pres'], marker='.', markersize=1, label="3D PAWS bmp2_pres")
    #     plt.plot(tsms_group.index, tsms_group['actual_pressure'], marker='.', markersize=1, label='TSMS')

    #     plt.title(f'{station_directories[i][8:14]} Pressure for {year_month}: 3D PAWS bmp2_pres versus TSMS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Pressure (mbar)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(data_destination+station_directories[i]+f"actual_pressure/{station_directories[i][8:14]}_bmp2_pres_{year_month}.png")
    #     #plt.savefig(data_destination+station_directories[i]+f"actual_pressure\\{station_directories[i][8:14]}_bmp2_pres_{year_month}.png")
        
    #     plt.clf()
    #     plt.close()

    
    # # Sea Level Pressure --------------------------------------------------------------------------------------------------------
    # print("Sea Level Pressure per sensor")
    # for year_month, paws_group in paws_df_FILTERED.groupby("year_month"):
    #     tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

    #     plt.figure(figsize=(12, 6))

    #     plt.plot(paws_group.index, paws_group['bmp2_slp'], marker='.', markersize=1, label="3D PAWS bmp2_slp")
    #     plt.plot(tsms_group.index, tsms_group['sea_level_pressure'], marker='.', markersize=1, label='TSMS')

    #     plt.title(f'{station_directories[i][8:14]} Sea Level Pressure for {year_month}: 3D PAWS bmp2_slp versus TSMS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Pressure (mbar)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(data_destination+station_directories[i]+f"sea_level_pressure/{station_directories[i][8:14]}_bmp2_slp_{year_month}.png")
    #     #plt.savefig(data_destination+station_directories[i]+f"sea_level_pressure\\{station_directories[i][8:14]}_bmp2_slp_{year_month}.png")
        
    #     plt.clf()
    #     plt.close()


    """
    =============================================================================================================================
    Create bar charts for monthly 3D PAWS rainfall accumulation compared to TSMS rainfall accumulation. COMPLETE RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: " \
    #         "Bar charts for rainfall accumulation -- complete records")
    
    # paws_df_FILTERED_2 = paws_df_FILTERED[paws_df_FILTERED['tipping'] >= 0].copy().reset_index()
    # tsms_df_FILTERED_2 = tsms_df_FILTERED[(tsms_df_FILTERED['total_rainfall']) >= 0].copy().reset_index()

    # tsms_df_REDUCED = tsms_df_FILTERED_2[ # need to filter TSMS s.t. data timeframe equals PAWS
    #     (tsms_df_FILTERED_2['date'] >= paws_df_FILTERED_2['date'].iloc[0]) & \
    #     (tsms_df_FILTERED_2['date'] <= paws_df_FILTERED_2['date'].iloc[-1])
    # ]

    # paws_totals = {}
    # tsms_totals = {}

    # for year_month, paws_grouped in paws_df_FILTERED_2.groupby("year_month"):
    #     tsms_grouped = tsms_df_REDUCED[tsms_df_REDUCED['year_month'] == year_month]

    #     merged_df = pd.merge(paws_grouped, tsms_grouped, on='date', how='inner')  # to eliminate bias

    #     merged_df['cumulative_rainfall_3DPAWS'] = merged_df['tipping'].cumsum()
    #     merged_df['cumulative_rainfall_TSMS'] = merged_df['total_rainfall'].cumsum()

    #     paws_total_rainfall = merged_df['cumulative_rainfall_3DPAWS'].iloc[-1] - merged_df['cumulative_rainfall_3DPAWS'].iloc[0]
    #     tsms_total_rainfall = merged_df['cumulative_rainfall_TSMS'].iloc[-1] - merged_df['cumulative_rainfall_TSMS'].iloc[0]

    #     paws_totals[year_month] = paws_total_rainfall
    #     tsms_totals[year_month] = tsms_total_rainfall
    
    # months = list(paws_totals.keys())
    # paws_values = list(paws_totals.values())
    # tsms_values = list(tsms_totals.values())

    # index = range(len(months))

    # plt.figure(figsize=(18, 12))

    # bars1 = plt.bar(index, paws_values, width=0.35, color='blue', label='3DPAWS Rainfall')
    # bars2 = plt.bar([i + 0.35 for i in index], tsms_values, width=0.35, color='orange', label='TSMS Rainfall')

    # # Add labels, title, and legend
    # plt.xlabel('Month')
    # plt.ylabel('Cumulative Rainfall (mm)')
    # plt.title(f'{station_directories[i][8:14]} Monthly Rainfall Comparison: 3DPAWS vs TSMS')
    # plt.xticks([i + 0.35 / 2 for i in index], [str(month) for month in months], rotation=45)
    # plt.legend()

    # # Add numerical values above bars
    # for bar in bars1:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=12)

    # for bar in bars2:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=12)

    # plt.tight_layout()

    # plt.savefig(data_destination+station_directories[i]+f"total_rainfall/{station_directories[i][8:14]}_monthly_rainfall.png")   
    # #plt.savefig(data_destination+station_directories[i]+f"total_rainfall\\{station_directories[i][8:14]}_monthly_rainfall.png") 
    
    # plt.clf()
    # plt.close()   

    """
    =============================================================================================================================
    Create bar charts for daily 3D PAWS rainfall accumulation (per instrument) compared to TSMS rainfall accumulation. MONTHLY RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: " \
    #                     "Bar charts for rainfall accumulation -- monthly records [INSTRUMENT VS REFERENCE COMPARISON]")
        
    # for year_month, paws_grouped in paws_df_FILTERED.groupby("year_month"):
    #     tsms_grouped = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

    #     merged_df = pd.merge(paws_grouped, tsms_grouped, on='date', how='inner')  # to eliminate bias

    #     # Calculate daily rainfall totals instead of cumulative
    #     merged_df['daily_rainfall_3DPAWS'] = merged_df['tipping']
    #     merged_df['daily_rainfall_TSMS'] = merged_df['total_rainfall']

    #     # Sum daily rainfall by date
    #     daily_totals = merged_df.groupby('year_month_day_x')[['daily_rainfall_3DPAWS', 'daily_rainfall_TSMS']].sum().reset_index()

    #     days = daily_totals['year_month_day_x'].dt.day
    #     paws_values = daily_totals['daily_rainfall_3DPAWS']
    #     tsms_values = daily_totals['daily_rainfall_TSMS']

    #     index = range(len(days))

    #     plt.figure(figsize=(20, 12))

    #     bars1 = plt.bar(index, paws_values, width=0.35, color='blue', label='3DPAWS Daily Rainfall')
    #     bars2 = plt.bar([i + 0.35 for i in index], tsms_values, width=0.35, color='orange', label='TSMS Daily Rainfall')

    #     # Add labels, title, and legend
    #     plt.xlabel(f'Day in {year_month}')
    #     plt.ylabel('Daily Rainfall (mm)')
    #     plt.title(f'{station_directories[i][8:14]} Daily Rainfall Comparison: 3DPAWS vs TSMS for {year_month}')
    #     plt.xticks([i + 0.35 / 2 for i in index], days, rotation=45)
    #     plt.ylim(0, 50)
    #     plt.legend()

    #     # Add numerical values above bars
    #     for bar in bars1:
    #         yval = bar.get_height()
    #         plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom', fontsize=12)

    #     for bar in bars2:
    #         yval = bar.get_height()
    #         plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom', fontsize=12)

    #     # Display the plot
    #     plt.tight_layout()

    #     # Save the plot
    #     # plt.savefig(data_destination + station_directories[i] + f"total_rainfall/raw/{station_directories[i][8:14]}_{year_month}_daily_rainfall.png")  
    #     plt.savefig(data_destination+station_directories[i]+f"total_rainfall\\{station_directories[i][8:14]}_{year_month}_daily_rainfall.png")    
        
    #     plt.clf()
    #     plt.close()


    """
    =============================================================================================================================
    Create wind rose plots of the 3D PAWS station data as well as the TSMS reference station. COMPLETE RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: Wind roses.")

    # # 3D PAWS  ------------------------------------------------------------------------------------------------------------------
    # print("3D PAWS")
    # paws_df_FILTERED.reset_index(inplace=True)
    # paws_df_FILTERED['date'] = pd.to_datetime(paws_df_FILTERED['date'])

    # # Convert the 'wind_speed' column to numeric, forcing errors to NaN
    # paws_df_FILTERED['wind_speed'] = pd.to_numeric(paws_df_FILTERED['wind_speed'], errors='coerce')

    # # Filter out rows where 'wind_speed' is NaN
    # paws_df_FILTERED_2 = paws_df_FILTERED[
    #     ((paws_df_FILTERED["wind_speed"].notna()) & (paws_df_FILTERED["wind_speed"] >= 0))
    # ]
    # paws_df_FILTERED_2 = paws_df_FILTERED_2[paws_df_FILTERED_2["wind_speed"] >= 3.0] # filter out variable winds 6 knots
    
    # paws_df_FILTERED_2.set_index('date', inplace=True)

    # # wind_speed_bins = [0, 2.0, 4.0, 6.0, 8.0, 10.0]       # for variable winds
    # # labels = ['0-2.0 m/s', '2.0-4.0 m/s', '4.0-6.0 m/s', '6.0-8.0 m/s', '8.0-10.0 m/s']
    # wind_speed_bins = [2.0, 4.0, 6.0, 8.0, 10.0]         # for non-variable winds
    # labels = [f"{wind_speed_bins[i]}–{wind_speed_bins[i+1]} m/s" for i in range(len(wind_speed_bins)-1)]

    # first_timestamp = pd.to_datetime(paws_df_FILTERED_2.index[0])
    # last_timestamp = pd.to_datetime(paws_df_FILTERED_2.index[-1])
    # # first_timestamp = pd.Timestamp("2022-09-09 00:00") # for reproducing TSMS study period
    # # last_timestamp = pd.Timestamp("2023-02-24 12:42")

    # paws_df_FILTERED_2.loc[:, 'wind_speed_category'] = pd.cut(paws_df_FILTERED_2['wind_speed'], bins=wind_speed_bins, labels=labels, right=False)

    # ax = WindroseAxes.from_ax()
    # ax.bar(paws_df_FILTERED_2['wind_dir'], paws_df_FILTERED_2['wind_speed'], normed=True, opening=0.8, edgecolor='white', bins=wind_speed_bins)
    # ax.set_legend(title=f"3D-PAWS {station_directories[i][8:len(station_directories[i])-1]} Wind Rose (m/s)", labels=labels)

    # # ax.set_rmax(35)
    # # ax.set_yticks([7, 14, 21, 28, 35])
    # # ax.set_yticklabels(['7%', '14%', '21%', '28%', '35%']) # for variable winds
    # ax.set_rmax(80)
    # ax.set_yticks([16, 32, 48, 64, 80])
    # ax.set_yticklabels(['16%', '32%', '48%', '64%', '80%']) # for NON-VARIABLE winds
        
    # ax.grid(True, linewidth=0.5)  # Thinner grid lines can improve readability

    # #plt.savefig(data_destination+station_directories[i]+f"wind/{station_directories[i][8:14]}_nonvariable_winds.png")
    # plt.savefig(data_destination+station_directories[i]+f"wind\\3DPAWS_{station_directories[i][8:14]}_nonvariable_winds.png")
    # # plt.savefig(data_destination+station_directories[i]+f"wind\\3DPAWS_{station_directories[i][8:14]}_variable_winds.png")
    # plt.clf()
    # plt.close()

    
    # # TSMS ----------------------------------------------------------------------------------------------------------------------
    # print("TSMS")
    # tsms_df_FILTERED.reset_index(inplace=True)
    # tsms_df_FILTERED['date'] = pd.to_datetime(tsms_df_FILTERED['date'])

    # tsms_df_FILTERED_2 = tsms_df_FILTERED[
    #     ~((tsms_df_FILTERED['avg_wind_speed'] == 0.0) & (tsms_df_FILTERED['avg_wind_dir'] == 0.0)) # filter out 0.0 pair wind speed & dir
    # ] 
    # tsms_df_FILTERED_2 = tsms_df_FILTERED[tsms_df_FILTERED['avg_wind_speed'] >= 3.0] # filter out variable winds
    
    # tsms_df_FILTERED_2.set_index('date', inplace=True)

    # tsms_subset = tsms_df_FILTERED_2.loc[first_timestamp:last_timestamp]

    # tsms_subset.loc[:, 'wind_speed_category'] = pd.cut(tsms_subset['avg_wind_speed'], bins=wind_speed_bins, labels=labels, right=False)

    # ax = WindroseAxes.from_ax() # TSMS data was already cleaned
    # ax.bar(tsms_subset['avg_wind_dir'], tsms_subset['avg_wind_speed'], normed=True, opening=0.8, edgecolor='white', bins=wind_speed_bins)
    
    # if i in [0,1,2]: ax.set_legend(title=f"TSMS Reference at Ankara (m/s)", labels=labels)
    # elif i in [3,4,5]: ax.set_legend(title=f"TSMS Reference at Konya (m/s)", labels=labels)
    # else: ax.set_legend(title=f"TSMS Reference at Adana (m/s)", labels=labels)
    
    
    # # ax.set_rmax(35)
    # # ax.set_yticks([7, 14, 21, 28, 35])
    # # ax.set_yticklabels(['7%', '14%', '21%', '28%', '35%']) # for complete records
    # ax.set_rmax(80)
    # ax.set_yticks([16, 32, 48, 64, 80])
    # ax.set_yticklabels(['16%', '32%', '48%', '64%', '80%']) # for non-variable winds

    # ax.grid(True, linewidth=0.5)  # Thinner grid lines can improve readability

    # # if i in [0,1,2]: plt.savefig(data_destination+station_directories[i]+f"wind/raw/Ankara_nonvariable_winds.png")
    # # elif i in [3,4,5]: plt.savefig(data_destination+station_directories[i]+f"wind/raw/Konya_nonvariable_winds.png")
    # # else: plt.savefig(data_destination+station_directories[i]+f"wind/raw/Adana_nonvariable_winds.png")
    # plt.savefig(data_destination+station_directories[i]+f"wind\\TSMS-Reference_nonvariable_winds.png")
    # #plt.savefig(data_destination+station_directories[i]+f"wind\\TSMS-Reference_variable_winds.png")
    
    # plt.clf()
    # plt.close()


    """
    =============================================================================================================================
    Create wind rose plots of the 3D PAWS station data as well as the TSMS reference station. MONTHLY RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: Wind roses for 3D PAWS and TSMS -- monthly records")

    # # 3D PAWS -------------------------------------------------------------------------------------------------------------------
    # print("3D PAWS -- nulls filtered")
    # paws_df_FILTERED.reset_index(inplace=True)
    # paws_df_FILTERED['date'] = pd.to_datetime(paws_df_FILTERED['date'])

    # # Convert the 'wind_speed' column to numeric and filter out NaN's (and variable winds)
    # paws_df_FILTERED['wind_speed'] = pd.to_numeric(paws_df_FILTERED['wind_speed'], errors='coerce')
    # #paws_df_FILTERED = paws_df[(paws_df["wind_speed"] >= 0)]                   # for variable winds
    # paws_df_FILTERED_2 = paws_df_FILTERED[paws_df_FILTERED['wind_speed'] >= 3.0]  # for NON-variable winds
    # paws_df_FILTERED_2.set_index('date', inplace=True)

    # first_timestamp = paws_df_FILTERED_2.index[0]
    # last_timestamp = paws_df_FILTERED_2.index[-1]
    # # first_timestamp = pd.Timestamp("2022-09-09 00:00") # for replicating the TSMS intercomparison study
    # # last_timestamp = pd.Timestamp("2023-02-24 12:42")

    # # wind_speed_bins = [0, 2.0, 4.0, 6.0, 8.0, 10.0]
    # # labels = ['0-2.0 m/s', '2.0-4.0 m/s', '4.0-6.0 m/s', '6.0-8.0 m/s', '8.0-10.0 m/s']       # for variable winds
    # wind_speed_bins = [2.0, 4.0, 6.0, 8.0, 10.0]        
    # labels = [f"{wind_speed_bins[i]}–{wind_speed_bins[i+1]} m/s" for i in range(len(wind_speed_bins)-1)]                    # for NON-variable winds

    # paws_df_FILTERED_2.loc[:, 'wind_speed_category'] = pd.cut(paws_df_FILTERED_2['wind_speed'], bins=wind_speed_bins, labels=labels, right=False)

    # for year_month, paws_group in paws_df_FILTERED_2.groupby("year_month"): 
    #     plt.figure(figsize=(20, 12))

    #     ax = WindroseAxes.from_ax()
    #     ax.bar(paws_group['wind_dir'], paws_group['wind_speed'], normed=True, opening=0.8, edgecolor='white', bins=wind_speed_bins)
    #     ax.set_legend(title=f"3D-PAWS {station_directories[i][8:len(station_directories[i])-1]} for {year_month} (m/s)", labels=labels)
        
    #     # ax.set_rmax(60)
    #     # ax.set_yticks([12, 24, 36, 48, 60])
    #     # ax.set_yticklabels(['12%', '24%', '36%', '48%', '60%']) # for variable winds
    #     ax.set_rmax(100)
    #     ax.set_yticks([20, 40, 60, 80, 100])
    #     ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%']) # for NON-variable winds
            
    #     ax.grid(True, linewidth=0.5)  # Thinner grid lines can improve readability

    #     plt.savefig(data_destination+station_directories[i]+f"wind\\3DPAWS_{station_directories[i][8:14]}_{year_month}_3DPAWS_nonvariable_winds.png")
    #     plt.clf()
    #     plt.close()


    # # TSMS ----------------------------------------------------------------------------------------------------------------------
    # print("TSMS -- raw") # TSMS dataset was already cleaned
    # tsms_df_FILTERED.reset_index(inplace=True)
    # tsms_df_FILTERED['date'] = pd.to_datetime(tsms_df_FILTERED['date'])

    # # Filter out rows where wind speed and wind direction are both 0.0 (and variable winds)
    # tsms_df_FILTERED_2 = tsms_df_FILTERED[~((tsms_df_FILTERED['avg_wind_speed'] == 0.0) & (tsms_df_FILTERED['avg_wind_dir'] == 0.0))]
    # tsms_df_FILTERED_2 = tsms_df_FILTERED[tsms_df_FILTERED['avg_wind_speed'] >= 3.0]  # for NON-variable winds
    # tsms_df_FILTERED_2.set_index('date', inplace=True)

    # tsms_subset = tsms_df_FILTERED_2.loc[first_timestamp:last_timestamp]

    # tsms_subset.loc[:, 'wind_speed_category'] = pd.cut(tsms_subset['avg_wind_speed'], bins=wind_speed_bins, labels=labels, right=False)

    # for year_month, tsms_group in tsms_subset.groupby("year_month"):
    #     plt.figure(figsize=(20, 12))

    #     ax = WindroseAxes.from_ax()
    #     ax.bar(tsms_group['avg_wind_dir'], tsms_group['avg_wind_speed'], normed=True, opening=0.8, edgecolor='white', bins=wind_speed_bins)

    #     if i in [0,1,2]: ax.set_legend(title=f"TSMS Reference at Ankara for {year_month} (m/s)", labels=labels)
    #     elif i in [3,4,5]: ax.set_legend(title=f"TSMS Reference at Konya for {year_month} (m/s)", labels=labels)
    #     else: ax.set_legend(title=f"TSMS Reference at Adana for {year_month} (m/s)", labels=labels)
        
    #     # ax.set_rmax(60)
    #     # ax.set_yticks([12, 24, 36, 48, 60])
    #     # ax.set_yticklabels(['12%', '24%', '36%', '48%', '60%']) # for variable winds
    #     ax.set_rmax(100)
    #     ax.set_yticks([20, 40, 60, 80, 100])
    #     ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%']) # for NON-variable winds
            
    #     ax.grid(True, linewidth=0.5)  # Thinner grid lines can improve readability

    #     plt.savefig(data_destination+station_directories[i]+f"wind\\TSMS-Reference_{year_month}_nonvariable_winds.png")
        
    #     plt.clf()
    #     plt.close()


    """
    =============================================================================================================================
    Create time series plots of the 3 temperature sensors compared to TSMS. MONTHLY RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][:len(station_directories[i])-1]} Temperature Comparison")

    # tsms_df_FILTERED_2 = tsms_df_FILTERED[paws_df_FILTERED.index[0]:paws_df_FILTERED.index[-1]]

    # tsms_df_FILTERED_2.reset_index(inplace=True)
    # paws_df_FILTERED.reset_index(inplace=True)

    # tsms_df_FILTERED_2['date'] = pd.to_datetime(tsms_df_FILTERED_2['date'])
    # paws_df_FILTERED['date'] = pd.to_datetime(paws_df_FILTERED['date'])

    # for year_month, paws_group in paws_df_FILTERED.groupby('year_month'):
    #     tsms_group = tsms_df_FILTERED_2[tsms_df_FILTERED_2['year_month'] == year_month]

    #     plt.figure(figsize=(12, 6))

    #     plt.plot(paws_group['date'], paws_group["bmp2_temp"], marker='.', markersize=1, label=f"3D PAWS bmp2_temp")
    #     plt.plot(paws_group['date'], paws_group["htu_temp"], marker='.', markersize=1, label=f"3D PAWS htu_temp")
    #     plt.plot(paws_group['date'], paws_group["mcp9808"], marker='.', markersize=1, label=f"3D PAWS mcp9808")
    #     plt.plot(tsms_group['date'], tsms_group['temperature'], marker='.', markersize=1, label='TSMS')

    #     plt.title(f'{station_directories[i][8:14]} Temperature Comparison for {year_month}: 3D PAWS versus TSMS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Temperature (˚C)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(data_destination+station_directories[i]+f"temperature/{station_directories[i][8:14]}_temp_comparison_{year_month}.png")
    #     #plt.savefig(data_destination+station_directories[i]+f"temperature-comparison\\{station_directories[i][8:14]}_temp_comparison_{year_month}.png")

    #     plt.clf()
    #     plt.close()

    """
    =============================================================================================================================
    Create scatter plots for each 3D PAWS sensor compared to the TSMS reference data. COMPLETE RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: " \
    #                     "Scatter plots for temperature, humidity, actual pressure, sea level pressure, and rain\n" \
    #                     "3D PAWS vs TSMS")

    # for variable in station_variables: 
    #     print(f"{variable} -- 3D PAWS vs TSMS")

    #     if variable == 'wind': variable = 'avg_wind_speed'

    #     for v in variable_mapper[f'{variable}']:
    #         if v == 'bme2_hum': 
    #             continue

    #         if v == 'tipping': 
    #             merged_df = pd.merge(
    #                 paws_df_FILTERED[['date', f"{v}"]],
    #                 tsms_df_FILTERED[['date', f"{variable}"]],
    #                 on='date',
    #                 how='inner'
    #             )

    #             hourly_totals = (
    #                 merged_df
    #                 .set_index('date') 
    #                 .resample('h')     # group into 1-hour bins
    #                 [[v, variable]]    # PAWS vs TSMS cols
    #                 .sum()             # sum rainfall within each hour
    #                 .reset_index()
    #             )

    #             x = hourly_totals[v]
    #             y = hourly_totals[variable]

    #             mask = x.notna() & y.notna()
    #             x, y = x[mask], y[mask]

    #             plt.figure(figsize=(6, 6))

    #             plt.scatter(x, y, alpha=0.5, s=10)

    #             # Calculate and plot trend line
    #             m, b = np.polyfit(x, y, 1)

    #             x_line = np.linspace(x.min(), x.max(), 100)
    #             y_line = (m * x_line) + b

    #             plt.plot(x_line, y_line, color='red', linewidth=2, label='Trend line')

    #             # Calculate correlation coefficient
    #             corr_coef, _ = pearsonr(x, y)
    #             plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    #             # Calculate RMSE
    #             rmse = np.sqrt(mean_squared_error(y, x))
    #             textstr = f'Correlation: {corr_coef:.2f}\nRMSE: {rmse:.2f}'
    #             bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8)
    #             plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
    #                     verticalalignment='top', bbox=bbox_props)

    #             plt.title(f'{station_directories[i][8:14]} 3D PAWS Rainfall versus TSMS')
    #             plt.xlabel(f'3D PAWS Rainfall')
    #             plt.ylabel(f"TSMS {variable}")

    #             plt.legend()
    #             plt.grid(True)
    #             plt.tight_layout()

    #             plt.savefig(data_destination+station_directories[i]+f"{variable}/{station_directories[i][8:14]}_rainfall_scatter-plot.png")
    
    #             plt.clf()
    #             plt.close()

    #             continue


    #         merged_df = pd.merge(
    #             paws_df_FILTERED[['date', f"{v}"]],
    #             tsms_df_FILTERED[['date', f"{variable}"]],
    #             on='date',
    #             how='inner'
    #         )

    #         x = merged_df[f'{v}'].to_numpy()
    #         y = merged_df[f'{variable}'].to_numpy()

    #         mask = np.isfinite(x) & np.isfinite(y)
    #         x, y = x[mask], y[mask]

    #         plt.figure(figsize=(6, 6))

    #         plt.scatter(x, y, alpha=0.5, s=10)

    #         # Calculate and plot trend line
    #         m, b = np.polyfit(x, y, 1)
    #         x_line = np.linspace(x.min(), x.max(), 100)
    #         y_line = (m * x_line) + b

    #         plt.plot(x_line, y_line, color='red', linewidth=2, label='Trend line')

    #         # Calculate correlation coefficient
    #         corr_coef, _ = pearsonr(x, y)
    #         plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    #         # Calculate RMSE
    #         rmse = np.sqrt(mean_squared_error(y, x))
    #         textstr = f'Correlation: {corr_coef:.2f}\nRMSE: {rmse:.2f}'
    #         bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8)
    #         plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
    #                 verticalalignment='top', bbox=bbox_props)

    #         plt.title(f'{station_directories[i][8:14]} 3D PAWS {v} versus TSMS')
    #         plt.xlabel(f'3D PAWS {v}')
    #         plt.ylabel(f"TSMS {variable}")

    #         plt.legend()
    #         plt.grid(True)
    #         plt.tight_layout()

    #         if variable == "avg_wind_speed": variable = "wind"

    #         plt.savefig(data_destination+station_directories[i]+f"{variable}/{station_directories[i][8:14]}_{v}_scatter-plot.png")

    #         plt.clf()
    #         plt.close()


    """
    =============================================================================================================================
    Create scatter plots for each 3D PAWS sensor compared to other 3D PAWS sensors (temp only). COMPLETE RECORDS
    ----> currently broke, skipping due to time constraints
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: " \
    #                     "Scatter plots for temperature -- complete records\n3D PAWS vs 3DPAWS")

    # merged_bmp_vs_htu = pd.merge(
    #     paws_df_FILTERED[['date', "bmp2_temp"]],
    #     paws_df_FILTERED[['date', 'htu_temp']],
    #     on='date',
    #     how='inner'
    # )
    # merged_bmp_vs_mcp = pd.merge(
    #     paws_df_FILTERED[['date', "bmp2_temp"]],
    #     paws_df_FILTERED[['date', 'mcp9808']],
    #     on='date',
    #     how='inner'
    # )
    # merged_htu_vs_mcp = pd.merge(
    #     paws_df_FILTERED[['date', "htu_temp"]],
    #     paws_df_FILTERED[['date', 'mcp9808']],
    #     on='date',
    #     how='inner'
    # )

    # plt.figure(figsize=(20, 12))

    # plt.scatter(merged_bmp_vs_htu["bmp2_temp"], merged_bmp_vs_htu["htu_temp"], alpha=0.5, s=10)

    # m, b = np.polyfit(merged_bmp_vs_htu["bmp2_temp"], merged_bmp_vs_htu["htu_temp"], 1)
    # plt.plot(merged_bmp_vs_htu["htu_temp"], m*merged_bmp_vs_htu["htu_temp"] + b, color='red', linewidth=2, label='Trend line')

    # corr_coef, _ = pearsonr(merged_bmp_vs_htu["bmp2_temp"], merged_bmp_vs_htu["htu_temp"])
    # plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # rmse = np.sqrt(mean_squared_error(merged_bmp_vs_htu["bmp2_temp"], merged_bmp_vs_htu["htu_temp"]))
    # textstr = f'Correlation: {corr_coef:.2f}\nRMSE: {rmse:.2f}'
    # bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8)
    # plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
    #         verticalalignment='top', bbox=bbox_props)

    # plt.title(f'{station_directories[i][8:14]} -- 3D PAWS bmp2_temp versus 3D PAWS htu_temp')
    # plt.xlabel('3D PAWS bmp2_temp')
    # plt.ylabel('3D PAWS htu_temp')    

    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()  

    # plt.savefig(data_destination+station_directories[i]+f"temperature\\{station_directories[i][8:14]}_bmp_vs_htu_scatter.png")    

    # plt.clf()
    # plt.close()  



    # plt.figure(figsize=(20, 12))

    # plt.scatter(merged_bmp_vs_mcp["bmp2_temp"], merged_bmp_vs_mcp["mcp9808"], alpha=0.5, s=10)

    # m, b = np.polyfit(merged_bmp_vs_mcp["bmp2_temp"], merged_bmp_vs_mcp["mcp9808"], 1)
    # plt.plot(merged_bmp_vs_mcp["bmp2_temp"], m*merged_bmp_vs_mcp["bmp2_temp"] + b, color='red', linewidth=2, label='Trend line')

    # corr_coef, _ = pearsonr(merged_bmp_vs_mcp["bmp2_temp"], merged_bmp_vs_mcp["mcp9808"])
    # plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # rmse = np.sqrt(mean_squared_error(merged_bmp_vs_mcp["bmp2_temp"], merged_bmp_vs_mcp["mcp9808"]))
    # textstr = f'Correlation: {corr_coef:.2f}\nRMSE: {rmse:.2f}'
    # bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8)
    # plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
    #         verticalalignment='top', bbox=bbox_props)

    # plt.title(f'{station_directories[i][8:14]} -- 3D PAWS bmp2_temp versus 3D PAWS mcp9808')
    # plt.xlabel('3D PAWS bmp2_temp')
    # plt.ylabel('3D PAWS mcp9808')    

    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()  

    # plt.savefig(data_destination+station_directories[i]+f"temperature\\{station_directories[i][8:14]}_bmp_vs_mcp9808_scatter.png")    

    # plt.clf()
    # plt.close()  



    # plt.figure(figsize=(20, 12))

    # plt.scatter(merged_htu_vs_mcp["mcp9808"], merged_htu_vs_mcp["htu_temp"], alpha=0.5, s=10)

    # m, b = np.polyfit(merged_htu_vs_mcp["mcp9808"], merged_htu_vs_mcp["htu_temp"], 1)
    # plt.plot(merged_htu_vs_mcp["mcp9808"], m*merged_htu_vs_mcp["mcp9808"] + b, color='red', linewidth=2, label='Trend line')

    # corr_coef, _ = pearsonr(merged_htu_vs_mcp["mcp9808"], merged_htu_vs_mcp["htu_temp"])
    # plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # rmse = np.sqrt(mean_squared_error(merged_htu_vs_mcp["mcp9808"], merged_htu_vs_mcp["htu_temp"]))
    # textstr = f'Correlation: {corr_coef:.2f}\nRMSE: {rmse:.2f}'
    # bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8)
    # plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
    #         verticalalignment='top', bbox=bbox_props)

    # plt.title(f'{station_directories[i][8:14]} -- 3D PAWS mcp9808 versus 3D PAWS htu_temp')
    # plt.xlabel('3D PAWS mcp9808')
    # plt.ylabel('3D PAWS htu_temp')    

    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()  

    # plt.savefig(data_destination+station_directories[i]+f"temperature\\{station_directories[i][8:14]}_mcp9808_vs_htu_scatter.png")    

    # plt.clf()
    # plt.close()  


    """
    =============================================================================================================================
    Creating box plots for temperature, relative humidity, pressure, wind speed, and rain. COMPLETE RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]} Box Plots")

    # paws_temp_cols = ['bmp2_temp', 'htu_temp', 'mcp9808']
    # paws_hum_cols = ['htu_hum']
    # paws_pres_cols = ['bmp2_pres']
    # paws_wind_cols = ['wind_speed']
    # paws_rain_cols = ['tipping']
    # tsms_temp_cols = ['temperature']
    # tsms_hum_cols = ['humidity']
    # tsms_pres_cols = ['actual_pressure']
    # tsms_wind_cols = ['avg_wind_speed']
    # tsms_rain_cols = ['total_rainfall']

    # combined_df = paws_df_FILTERED.merge(
    #     tsms_df_FILTERED[tsms_temp_cols + tsms_hum_cols + tsms_pres_cols + tsms_wind_cols + tsms_rain_cols], 
    #     left_index=True, right_index=True
    # )
    # combined_temp_df = combined_df[paws_temp_cols + ['temperature']]
    # combined_hum_df = combined_df[paws_hum_cols + ['humidity']]
    # combined_pres_df = combined_df[paws_pres_cols + ['actual_pressure']]
    # combined_wind_df = combined_df[paws_wind_cols + ['avg_wind_speed']]
    # combined_rain_df = combined_df[paws_rain_cols + ['total_rainfall']]

    # # temperature box plot
    # plt.figure(figsize=(12, 12))
    # sns.boxplot(data=combined_temp_df, palette=['blue', 'orange', 'green', 'red'])

    # plt.title(f'{station_directories[i][8:14]} Temperature Box Plot: 3D-PAWS & TSMS')
    # plt.xlabel('Temperature Sensors')
    # plt.ylabel('Temperature (°C)')
    # plt.xticks([0, 1, 2, 3], ['BMP2 Temp', 'HTU Temp', 'MCP9808 Temp', 'TSMS Reference Temp'])
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"/temperature/{station_directories[i][8:14]}_box-plot.png")
    # plt.clf()
    # plt.close()

    # # humidity boxplot
    # plt.figure(figsize=(12, 12))
    # sns.boxplot(data=combined_hum_df, palette=['blue', 'red'])

    # plt.title(f'{station_directories[i][8:14]} Humidity Box Plot: 3D-PAWS & TSMS')
    # plt.xlabel('Humidity Sensors')
    # plt.ylabel('Humidity (%)')
    # plt.xticks([0, 1], ['HTU Hum', 'TSMS Reference Hum'])
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"/humidity/{station_directories[i][8:14]}_box-plot.png")
    # plt.clf()
    # plt.close()

    # # pressure box plots
    # plt.figure(figsize=(12, 12))
    # sns.boxplot(data=combined_pres_df, palette=['blue', 'red'])

    # plt.title(f'{station_directories[i][8:14]} Pressure Box Plot: 3D-PAWS & TSMS')
    # plt.xlabel('Pressure Sensors')
    # plt.ylabel('Station Pressure (hPa)')
    # plt.xticks([0, 1], ['BMP2 Pressure', 'TSMS Reference Pressure'])
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"/actual_pressure/{station_directories[i][8:14]}_box-plot.png")
    # plt.clf()
    # plt.close()

    # # wind speed box plots
    # plt.figure(figsize=(12, 12))
    # sns.boxplot(data=combined_pres_df, palette=['blue', 'red'])

    # plt.title(f'{station_directories[i][8:14]} Wind Speed Box Plot: 3D-PAWS & TSMS')
    # plt.xlabel('Anemometers')
    # plt.ylabel('Wind Speed (m/s)')
    # plt.xticks([0, 1], ['3D-PAWS Anemometer', 'TSMS Reference Anemometer'])
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"/wind/{station_directories[i][8:14]}_box-plot.png")
    # plt.clf()
    # plt.close()

    # # precipitation box plots
    # plt.figure(figsize=(12, 12))
    # sns.boxplot(data=combined_pres_df, palette=['blue', 'red'])

    # plt.title(f'{station_directories[i][8:14]} Precipitation Box Plot: 3D-PAWS & TSMS')
    # plt.xlabel('Precipitation Sensors')
    # plt.ylabel('Total Rainfall (mm)')
    # plt.xticks([0, 1], ['3D-PAWS Tipping Bucket', 'TSMS Reference Gauge'])
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"/total_rainfall/{station_directories[i][8:14]}_box-plot.png")
    # plt.clf()
    # plt.close()


    """
    =============================================================================================================================
    Violin plots for temperature, humidity, and pressure. COMPLETE RECORDS
    * this style of plot is uninformative for wind speed and precip
    =============================================================================================================================
    """
    print(f"{station_directories[i][8:14]} Violin Plots")

    y_axis_labels = {
        'temperature': 'Temperature (˚C)',
        'humidity': 'Relative Humidity (%)',
        'actual_pressure': 'Pressure (hPa)'
    }

    for var in station_variables:
        if var in ['sea_level_pressure', 'wind', 'total_rainfall']: continue

        for v in variable_mapper[var]:
            if v in ['bme2_hum']: continue

            x_axis_labels = {
                v: f'3D-PAWS ({v})',
                var: 'TSMS Reference Sensor'
            }
            custom_palette = {
                x_axis_labels[v]: "#1f77b4",        # Blue
                "TSMS Reference Sensor": "#d62728"  # Red
            }

            merged_df = pd.merge(
                paws_df_FILTERED.reset_index()[['date', v]], 
                tsms_df_FILTERED.reset_index()[['date' ,var]],
                on='date', 
                suffixes=['_paws', '_tsms']
            )
        
            long_df = pd.melt(  # Seaborn wants data in long format
                merged_df,
                id_vars='date',
                value_vars=[f'{v}', f'{var}'],
                var_name='Source',
                value_name='Value'
            )
            long_df['Source'] = long_df['Source'].map(x_axis_labels)

            if v in ['htu_temp', 'htu_hum']:    # truncate to separate htu from sht
                dates = [
                    pd.to_datetime("2024-01-17 12:00:00"),  # Ankara
                    pd.to_datetime("2024-01-12 12:00:00"),  # Konya
                    pd.to_datetime("2024-01-15 12:00:00")   # Adana
                ]

                htu_df, sht_df = None, None
                if i in [0,1,2]: 
                    htu_df = long_df[((long_df['date'] >= long_df['date'].iloc[0]) & (long_df['date'] <= dates[0]))]
                    sht_df = long_df[((long_df['date'] > dates[0]) & (long_df['date'] <= long_df['date'].iloc[-1]))]
                elif i in [3,4,5]: 
                    htu_df = long_df[((long_df['date'] >= long_df['date'].iloc[0]) & (long_df['date'] <= dates[1]))]
                    sht_df = long_df[((long_df['date'] > dates[1]) & (long_df['date'] <= long_df['date'].iloc[-1]))]
                else: 
                    htu_df = long_df[((long_df['date'] >= long_df['date'].iloc[0]) & (long_df['date'] <= dates[2]))]
                    sht_df = long_df[((long_df['date'] > dates[2]) & (long_df['date'] <= long_df['date'].iloc[-1]))]

                plt.figure(figsize=(12, 12))
                sns.violinplot(x='Source', y='Value', data=htu_df, palette=custom_palette, inner='box')
                plt.title(f"{station_directories[i][8:14]} Violin Plot of {var[0].upper()+var[1:]}: 3D PAWS ({v}) vs TSMS Reference Sensor")
                plt.ylabel(y_axis_labels[var])
                plt.tight_layout()
                plt.savefig(data_destination+station_directories[i]+f"/{var}/{station_directories[i][8:14]}_htu_violin-plot.png")
                plt.clf()
                plt.close()

                plt.figure(figsize=(12, 12))
                sns.violinplot(x='Source', y='Value', data=sht_df, palette=custom_palette, inner='box')
                plt.title(f"{station_directories[i][8:14]} Violin Plot of {var[0].upper()+var[1:]}: 3D PAWS (sht) vs TSMS Reference Sensor")
                plt.ylabel(y_axis_labels[var])
                plt.tight_layout()
                plt.savefig(data_destination+station_directories[i]+f"/{var}/{station_directories[i][8:14]}_sht_violin-plot.png")
                plt.clf()
                plt.close()

                continue

            plt.figure(figsize=(12, 12))
            sns.violinplot(x='Source', y='Value', data=long_df, palette=custom_palette, inner='box')
            plt.title(f"{station_directories[i][8:14]} Violin Plot of {var[0].upper()+var[1:]}: 3D PAWS ({v}) vs TSMS Reference Sensor")
            plt.ylabel(y_axis_labels[var])
            plt.tight_layout()
            plt.savefig(data_destination+station_directories[i]+f"/{var}/{station_directories[i][8:14]}_{v}_violin-plot.png")
            plt.clf()
            plt.close()


    """
    =============================================================================================================================
    Creating histograms for each variable. COMPLETE RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]} Histograms")

    # for variable in variable_mapper.keys():
    #     if variable in ['sea_level_pressure', 'total_rainfall']: continue

    #     for var in variable_mapper[variable]:
    #         if var in ['bme_hum']: continue

    #         merged_df = pd.merge(
    #             paws_df_FILTERED.reset_index()[['date', var]], 
    #             tsms_df_FILTERED.reset_index()[['date' ,variable]],
    #             on='date', 
    #             suffixes=['_paws', '_tsms']
    #         )

    #         cleaned_df = merged_df.dropna(subset=[var, variable])

    #         plt.figure(figsize=(12,12))
    #         plt.hist(cleaned_df[var], bins=15, edgecolor='black')
    #         plt.xlabel(variable[0].upper() + variable[1:])
    #         plt.ylabel("Frequency")
    #         plt.title(f"3D-PAWS {variable[0].upper()+variable[1:]} ({var})")
    #         plt.tight_layout()
    #         if variable in ['avg_wind_speed', 'avg_wind_dir']: 
    #             plt.savefig(data_destination+station_directories[i]+f"/wind/{station_directories[i][8:14]}_{variable}_histogram.png")
    #         else:
    #             plt.savefig(data_destination+station_directories[i]+f"/{variable}/{station_directories[i][8:14]}_{var}_histogram.png")
    #         plt.clf()
    #         plt.close()

    #         plt.figure(figsize=(12,12))
    #         plt.hist(cleaned_df[variable], bins=15, edgecolor='black', color='indianred')
    #         plt.xlabel(variable[0].upper()+variable[1:])
    #         plt.ylabel("Frequency")
    #         plt.title(f"TSMS {variable[0].upper()+variable[1:]}")
    #         plt.tight_layout()
    #         if variable in ['avg_wind_speed', 'avg_wind_dir']:
    #             if i in [0,1,2]:
    #                 plt.savefig(data_destination+station_directories[i]+f"/wind/TSMS-Ankara_{variable}_histogram.png")
    #             elif i in [3,4,5]:
    #                 plt.savefig(data_destination+station_directories[i]+f"/wind/TSMS-Konya_{variable}_histogram.png")
    #             else:
    #                 plt.savefig(data_destination+station_directories[i]+f"/wind/TSMS-Adana_{variable}_histogram.png")
    #         else:
    #             if i in [0,1,2]:
    #                 plt.savefig(data_destination+station_directories[i]+f"/{variable}/TSMS-Ankara_histogram.png")
    #             elif i in [3,4,5]:
    #                 plt.savefig(data_destination+station_directories[i]+f"/{variable}/TSMS-Konya_histogram.png")
    #             else:
    #                 plt.savefig(data_destination+station_directories[i]+f"/{variable}/TSMS-Adana_histogram.png")
    #         plt.clf()
    #         plt.close()

"""
=============================================================================================================================
Site-wide comparison of each type of temperature sensor versus the reference. MONTHLY RECORDS
=============================================================================================================================
"""
# print("Temperature comparison (this will take some time)")
# sites = {
#     0:"Ankara", 1:"Konya", 2:"Adana"
# }
# station_map = {
#     0:[0,1,2], 1:[3,4,5], 2:[6,7,8]
# }

# for i in range(3):
#     print(f"\t{sites[i]}: Temperature Comparison per Sensor")

#     inst_1 = paws_dfs[station_map[i][0]].copy(deep=True)
#     inst_2 = paws_dfs[station_map[i][1]].copy(deep=True)
#     inst_3 = paws_dfs[station_map[i][2]].copy(deep=True)
#     tsms_ref = tsms_dfs[i*3].copy(deep=True)

#     inst_1.reset_index(inplace=True)
#     inst_2.reset_index(inplace=True)
#     inst_3.reset_index(inplace=True)
#     tsms_ref.reset_index(inplace=True)

#     for year_month in set(inst_1['year_month']) & \
#                         set(inst_2['year_month']) & \
#                             set(inst_3['year_month']) & \
#                                 set(tsms_ref['year_month']):
#         inst_1_grouped = inst_1[inst_1['year_month'] == year_month]
#         inst_2_grouped = inst_2[inst_2['year_month'] == year_month]
#         inst_3_grouped = inst_3[inst_3['year_month'] == year_month]
#         tsms_grouped = tsms_ref[tsms_ref['year_month'] == year_month]

#         merged_df = pd.merge(inst_1_grouped, inst_2_grouped, on='date', suffixes=('_1', '_2'))
#         merged_df = pd.merge(merged_df, inst_3_grouped, on='date', suffixes=('', '_3'))
#         merged_df = pd.merge(merged_df, tsms_grouped, on='date', suffixes=('', '_tsms'))

#         for sensor in variable_mapper['temperature']:
#             plt.figure(figsize=(20, 12))

#             plt.plot(merged_df['date'], merged_df[f'{sensor}_1'], marker='.', markersize=1, label=f"TSMS0{station_map[i][0]} {sensor}")
#             plt.plot(merged_df['date'], merged_df[f'{sensor}_2'], marker='.', markersize=1, label=f"TSMS0{station_map[i][1]} {sensor}")
#             plt.plot(merged_df['date'], merged_df[f'{sensor}'], marker='.', markersize=1, label=f"TSMS0{station_map[i][2]} {sensor}")
#             plt.plot(merged_df['date'], merged_df[f'temperature'], marker='.', markersize=1, label=f'{sites[i]} TSMS Reference')

#             plt.title(f'{sites[i]} {year_month}: {sensor} Temperature Comparison')
#             plt.xlabel('Date')
#             plt.ylabel('Temperature (˚C)')
#             plt.xticks(rotation=45)

#             plt.legend()

#             plt.grid(True)
#             plt.tight_layout()

#             #print(data_destination+station_directories[i]+f"temperature/{sites[i]}_{sensor}_{year_month}_temperature_comparison.png")
#             plt.savefig(data_destination+station_directories[i]+f"temperature/{sites[i]}_{sensor}_{year_month}_temperature_comparison.png")
#             #plt.savefig(data_destination+sites[i]+f"\\temperature\\{sites[i]}_{sensor}_{year_month}_temperature_comparison.png")

#             plt.clf()
#             plt.close()


"""
=============================================================================================================================
Site-wide comparison of each type of humidity sensor versus the reference. MONTHLY RECORDS
=============================================================================================================================
"""
# print("Humidity comparison (this will take some time)")
# sites = {
#     0:"Ankara", 1:"Konya", 2:"Adana"
# }
# station_map = {
#     0:[0,1,2], 1:[3,4,5], 2:[6,7,8]
# }

# for i in range(3):
#     print(f"\t{sites[i]}: Humidity Comparison per Sensor")

#     inst_1 = paws_dfs[station_map[i][0]].copy(deep=True)
#     inst_2 = paws_dfs[station_map[i][1]].copy(deep=True)
#     inst_3 = paws_dfs[station_map[i][2]].copy(deep=True)
#     tsms_ref = tsms_dfs[i*3].copy(deep=True)

#     inst_1.reset_index(inplace=True)
#     inst_2.reset_index(inplace=True)
#     inst_3.reset_index(inplace=True)
#     tsms_ref.reset_index(inplace=True)

#     for year_month in set(inst_1['year_month']) & \
#                         set(inst_2['year_month']) & \
#                             set(inst_3['year_month']) & \
#                                 set(tsms_ref['year_month']):
#         inst_1_grouped = inst_1[inst_1['year_month'] == year_month]
#         inst_2_grouped = inst_2[inst_2['year_month'] == year_month]
#         inst_3_grouped = inst_3[inst_3['year_month'] == year_month]
#         tsms_grouped = tsms_ref[tsms_ref['year_month'] == year_month]

#         merged_df = pd.merge(inst_1_grouped, inst_2_grouped, on='date', suffixes=('_1', '_2'))
#         merged_df = pd.merge(merged_df, inst_3_grouped, on='date', suffixes=('', '_3'))
#         merged_df = pd.merge(merged_df, tsms_grouped, on='date', suffixes=('', '_tsms'))

#         for sensor in variable_mapper['humidity']:
#             if sensor == 'bme2_hum': continue

#             plt.figure(figsize=(20, 12))

#             plt.plot(merged_df['date'], merged_df[f'{sensor}_1'], marker='.', markersize=1, label=f"TSMS0{station_map[i][0]} {sensor}")
#             plt.plot(merged_df['date'], merged_df[f'{sensor}_2'], marker='.', markersize=1, label=f"TSMS0{station_map[i][1]} {sensor}")
#             plt.plot(merged_df['date'], merged_df[f'{sensor}'], marker='.', markersize=1, label=f"TSMS0{station_map[i][2]} {sensor}")
#             plt.plot(merged_df['date'], merged_df[f'humidity'], marker='.', markersize=1, label=f'{sites[i]} TSMS Reference')

#             plt.title(f'{sites[i]} {year_month}: {sensor} Relative Humidity Comparison')
#             plt.xlabel('Date')
#             plt.ylabel('Relative Humidity (%)')
#             plt.xticks(rotation=45)

#             plt.legend()

#             plt.grid(True)
#             plt.tight_layout()

#             #plt.savefig(data_destination+station_directories[i]+f"total_rainfall/raw/{station_directories[i][8:14]}_rainfall_accumulation_TEST.png")
#             plt.savefig(data_destination+sites[i]+f"\\humidity\\{sites[i]}_{sensor}_{year_month}_humidity_comparison.png")

#             plt.clf()
#             plt.close()


"""
=============================================================================================================================
Site-wide comparison of each type of pressure sensor versus the reference. MONTHLY RECORDS
=============================================================================================================================
"""
# sites = {
#     0:"Ankara", 1:"Konya", 2:"Adana"
# }
# station_map = {
#     0:[0,1,2], 1:[3,4,5], 2:[6,7,8]
# }

# for i in range(3):
#     print(f"\t{sites[i]}: Actual & sea level pressure [PER SITE]")

#     inst_1 = paws_dfs[station_map[i][0]].copy(deep=True)
#     inst_2 = paws_dfs[station_map[i][1]].copy(deep=True)
#     inst_3 = paws_dfs[station_map[i][2]].copy(deep=True)
#     tsms_ref = tsms_dfs[i*3].copy(deep=True)

#     inst_1.reset_index(inplace=True)
#     inst_2.reset_index(inplace=True)
#     inst_3.reset_index(inplace=True)
#     tsms_ref.reset_index(inplace=True)

#     for year_month in set(inst_1['year_month']) & \
#                         set(inst_2['year_month']) & \
#                             set(inst_3['year_month']) & \
#                                 set(tsms_ref['year_month']):
#         inst_1_grouped = inst_1[inst_1['year_month'] == year_month]
#         inst_2_grouped = inst_2[inst_2['year_month'] == year_month]
#         inst_3_grouped = inst_3[inst_3['year_month'] == year_month]
#         tsms_grouped = tsms_ref[tsms_ref['year_month'] == year_month]

#         merged_df = pd.merge(inst_1_grouped, inst_2_grouped, on='date', suffixes=('_1', '_2'))
#         merged_df = pd.merge(merged_df, inst_3_grouped, on='date', suffixes=('', '_3'))
#         merged_df = pd.merge(merged_df, tsms_grouped, on='date', suffixes=('', '_tsms'))

#         plt.figure(figsize=(20, 12))

#         plt.plot(merged_df['date'], merged_df['bmp2_pres_1'], marker='.', markersize=1, label=f"TSMS0{station_map[i][0]} 3D PAWS")
#         plt.plot(merged_df['date'], merged_df['bmp2_pres_2'], marker='.', markersize=1, label=f"TSMS0{station_map[i][1]} 3D PAWS")
#         plt.plot(merged_df['date'], merged_df['bmp2_pres'], marker='.', markersize=1, label=f"TSMS0{station_map[i][2]} 3D PAWS")
#         plt.plot(merged_df['date'], merged_df['actual_pressure'], marker='.', markersize=1, label=f'{sites[i]} TSMS Reference')

#         plt.title(f'{sites[i]} {year_month} Station Pressure')
#         plt.xlabel('Date')
#         plt.ylabel('Pressure (hPa)')
#         plt.xticks(rotation=45)

#         plt.legend()

#         plt.grid(True)
#         plt.tight_layout()

#         #plt.savefig(data_destination+station_directories[i]+f"total_rainfall/raw/{station_directories[i][8:14]}_rainfall_accumulation_TEST.png")
#         plt.savefig(data_destination+sites[i]+f"\\actual_pressure\\{sites[i]}_{year_month}_station_pressure.png")

#         plt.clf()
#         plt.close()

#         print("\t\tSea Level Pressure")
#         print(f"\t\t\t{sites[i]} at {year_month}")

#         plt.figure(figsize=(20, 12))

#         plt.plot(merged_df['date'], merged_df['bmp2_slp_1'], marker='.', markersize=1, label=f"TSMS0{station_map[i][0]} 3D PAWS")
#         plt.plot(merged_df['date'], merged_df['bmp2_slp_2'], marker='.', markersize=1, label=f"TSMS0{station_map[i][1]} 3D PAWS")
#         plt.plot(merged_df['date'], merged_df['bmp2_slp'], marker='.', markersize=1, label=f"TSMS0{station_map[i][2]} 3D PAWS")
#         plt.plot(merged_df['date'], merged_df['sea_level_pressure'], marker='.', markersize=1, label=f'{sites[i]} TSMS Reference')

#         plt.title(f'{sites[i]} {year_month} Sea Level Pressure')
#         plt.xlabel('Date')
#         plt.ylabel('Sea Level Pressure (hPa)')
#         plt.xticks(rotation=45)

#         plt.legend()

#         plt.grid(True)
#         plt.tight_layout()

#         #plt.savefig(data_destination+station_directories[i]+f"total_rainfall/raw/{station_directories[i][8:14]}_rainfall_accumulation_TEST.png")
#         plt.savefig(data_destination+sites[i]+f"\\sea_level_pressure\\{sites[i]}_{year_month}_station_sea_level_pressure.png")

#         plt.clf()
#         plt.close()


"""
=============================================================================================================================
Rainfall accumulation time series of each individual 3D PAWS station data versus the TSMS reference station. COMPLETE RECORDS
=============================================================================================================================
"""
# sites = {
#     0:"Ankara", 1:"Konya", 2:"Adana"
# }
# station_map = {
#     0:[0,1,2], 1:[3,4,5], 2:[6,7,8]
# }

# for i in range(3):
#     print(f"\t{sites[i]}: Rainfall accumulation [PER SITE]")

#     inst_1 = paws_dfs[station_map[i][0]].copy(deep=True)
#     inst_2 = paws_dfs[station_map[i][1]].copy(deep=True)
#     inst_3 = paws_dfs[station_map[i][2]].copy(deep=True)
#     tsms_ref = tsms_dfs[i*3].copy(deep=True)

#     inst_1.reset_index(inplace=True)
#     inst_2.reset_index(inplace=True)
#     inst_3.reset_index(inplace=True)
#     tsms_ref.reset_index(inplace=True)

#     merged_df = pd.merge(inst_1, inst_2, on='date', suffixes=('_1', '_2'))
#     merged_df = pd.merge(merged_df, inst_3, on='date', suffixes=('', '_3'))
#     merged_df = pd.merge(merged_df, tsms_ref, on='date', suffixes=('', '_tsms'))

#     merged_df['cumulative_rainfall_3DPAWS_1'] = merged_df['tipping_1'].cumsum()
#     merged_df['cumulative_rainfall_3DPAWS_2'] = merged_df['tipping_2'].cumsum()
#     merged_df['cumulative_rainfall_3DPAWS_3'] = merged_df['tipping'].cumsum()
#     merged_df['cumulative_rainfall_TSMS'] = merged_df['total_rainfall'].cumsum()

#     plt.figure(figsize=(20, 12))

#     plt.plot(merged_df['date'], merged_df['cumulative_rainfall_3DPAWS_1'], marker='.', markersize=1, label=f"TSMS0{station_map[i][0]} 3D PAWS")
#     plt.plot(merged_df['date'], merged_df['cumulative_rainfall_3DPAWS_2'], marker='.', markersize=1, label=f"TSMS0{station_map[i][1]} 3D PAWS")
#     plt.plot(merged_df['date'], merged_df['cumulative_rainfall_3DPAWS_3'], marker='.', markersize=1, label=f"TSMS0{station_map[i][2]} 3D PAWS")
#     plt.plot(merged_df['date'], merged_df['cumulative_rainfall_TSMS'], marker='.', markersize=1, label=f'{sites[i]} TSMS Reference')

#     plt.title(f'{sites[i]} Rainfall Accumulation')
#     plt.xlabel('Date')
#     plt.ylabel('Rainfall (mm)')
#     plt.xticks(rotation=45)

#     plt.legend()

#     plt.grid(True)
#     plt.tight_layout()

#     #plt.savefig(data_destination+station_directories[i]+f"total_rainfall/raw/{station_directories[i][8:14]}_rainfall_accumulation_TEST.png")
#     plt.savefig(data_destination+sites[i]+f"\\total_rainfall\\{sites[i]}_rainfall_accumulation.png")

#     plt.clf()
#     plt.close()


"""
=============================================================================================================================
Create bar charts for daily 3D PAWS rainfall accumulation (per site) compared to TSMS rainfall accumulation. MONTHLY RECORDS
=============================================================================================================================
"""
# sites = {
#     0:"Ankara", 1:"Konya", 2:"Adana"
# }
# station_map = {
#     0:[0,1,2], 1:[3,4,5], 2:[6,7,8]
# }

# for i in range(3):
#     print(f"{sites[i]}: Bar charts for rainfall accumulation -- daily totals [ALL INSTRUMENTS PER SITE]")

#     inst_1 = paws_dfs[station_map[i][0]].copy(deep=True)
#     inst_2 = paws_dfs[station_map[i][1]].copy(deep=True)
#     inst_3 = paws_dfs[station_map[i][2]].copy(deep=True)
#     tsms_ref = tsms_dfs[i*3].copy(deep=True)

#     inst_1.reset_index(inplace=True)
#     inst_2.reset_index(inplace=True)
#     inst_3.reset_index(inplace=True)
#     tsms_ref.reset_index(inplace=True)
        
#     for year_month in set(inst_1['year_month']) & \
#                         set(inst_2['year_month']) & \
#                             set(inst_3['year_month']) & \
#                                 set(tsms_ref['year_month']):
#         inst_1_grouped = inst_1[inst_1['year_month'] == year_month]
#         inst_2_grouped = inst_2[inst_2['year_month'] == year_month]
#         inst_3_grouped = inst_3[inst_3['year_month'] == year_month]
#         tsms_grouped = tsms_ref[tsms_ref['year_month'] == year_month]

#         merged_df = pd.merge(inst_1_grouped, inst_2_grouped, on='date', suffixes=('_1', '_2'))
#         merged_df = pd.merge(merged_df, inst_3_grouped, on='date', suffixes=('', '_3'))
#         merged_df = pd.merge(merged_df, tsms_grouped, on='date', suffixes=('', '_tsms'))

#         merged_df['daily_rainfall_1'] = merged_df['tipping_1']    # Calculate daily rainfall totals
#         merged_df['daily_rainfall_2'] = merged_df['tipping_2']
#         merged_df['daily_rainfall_3'] = merged_df['tipping']
#         merged_df['daily_rainfall_TSMS'] = merged_df['total_rainfall']
        
#         daily_totals = merged_df.groupby('year_month_day')[     # Sum daily rainfall by date
#             ['daily_rainfall_1', 'daily_rainfall_2', 'daily_rainfall_3', 'daily_rainfall_TSMS']
#         ].sum().reset_index()

#         days = daily_totals['year_month_day'].dt.day
#         values_1 = daily_totals['daily_rainfall_1']
#         values_2 = daily_totals['daily_rainfall_2']
#         values_3 = daily_totals['daily_rainfall_3']
#         values_tsms = daily_totals['daily_rainfall_TSMS']

#         index = range(len(days))

#         plt.figure(figsize=(20, 12), constrained_layout=True)

#         bar_width = 0.2
#         bars1 = plt.bar(index, values_1, width=bar_width, color='blue', label=f'3DPAWS TSMS0{station_map[i][0]}')
#         bars2 = plt.bar([i + bar_width for i in index], values_2, width=bar_width, color='orange', label=f'3DPAWS TSMS0{station_map[i][1]}')
#         bars3 = plt.bar([i + 2*bar_width for i in index], values_3, width=bar_width, color='green', label=f'3DPAWS TSMS0{station_map[i][2]}')
#         bars4 = plt.bar([i + 3*bar_width for i in index], values_tsms, width=bar_width, color='red', label='TSMS Reference')
        
#         plt.xlabel(f'Day in {year_month}', fontsize=8)          # Add labels, title, and legend
#         plt.ylabel('Daily Rainfall (mm)', fontsize=8)
#         plt.title(f'{sites[i]} Daily Rainfall Comparison: All Instruments for {year_month}', fontsize=8)
#         plt.xticks([i + 1.5*bar_width for i in index], days, rotation=45)
#         plt.ylim(0, 50)
#         plt.legend()

#         for bars in [bars1, bars2, bars3, bars4]:               # Add numerical values above bars
#             for bar in bars:
#                 yval = bar.get_height()
#                 plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom', fontsize=8)

#         plt.savefig(data_destination+sites[i]+f"\\total_rainfall\\{sites[i]}_{year_month}_daily_rainfall_all_instruments.png")    
        
#         plt.clf()
#         plt.close()

    
"""
=============================================================================================================================
Create bar charts for daily 3D PAWS rainfall accumulation (per site) compared to TSMS rainfall accumulation. COMPLETE RECORDS
=============================================================================================================================
"""
# sites = {
#     0:"Ankara", 1:"Konya", 2:"Adana"
# }
# station_map = {
#     0:[0,1,2], 1:[3,4,5], 2:[6,7,8]
# }

# for i in range(3):
#     print(f"{sites[i]}: Bar charts for rainfall accumulation -- monthly totals [ALL INSTRUMENTS PER SITE]")

#     inst_1 = paws_dfs[station_map[i][0]].copy(deep=True)
#     inst_2 = paws_dfs[station_map[i][1]].copy(deep=True)
#     inst_3 = paws_dfs[station_map[i][2]].copy(deep=True)
#     tsms_ref = tsms_dfs[i*3].copy(deep=True)

#     inst_1.reset_index(inplace=True)
#     inst_2.reset_index(inplace=True)
#     inst_3.reset_index(inplace=True)
#     tsms_ref.reset_index(inplace=True)

#     merged_df = pd.merge(inst_1, inst_2, on='date', suffixes=('_1', '_2'))
#     merged_df = pd.merge(merged_df, inst_3, on='date', suffixes=('', '_3'))
#     merged_df = pd.merge(merged_df, tsms_ref, on='date', suffixes=('', '_tsms'))

#     merged_df['monthly_rainfall_1'] = merged_df['tipping_1']
#     merged_df['monthly_rainfall_2'] = merged_df['tipping_2']
#     merged_df['monthly_rainfall_3'] = merged_df['tipping']
#     merged_df['monthly_rainfall_TSMS'] = merged_df['total_rainfall']

#     monthly_totals = merged_df.groupby('year_month')[
#         ['monthly_rainfall_1', 'monthly_rainfall_2', 'monthly_rainfall_3', 'monthly_rainfall_TSMS']
#     ].sum().reset_index()

#     months = monthly_totals['year_month']
#     values_1 = monthly_totals['monthly_rainfall_1']
#     values_2 = monthly_totals['monthly_rainfall_2']
#     values_3 = monthly_totals['monthly_rainfall_3']
#     values_tsms = monthly_totals['monthly_rainfall_TSMS']

#     index = range(len(months))

#     plt.figure(figsize=(20, 12), constrained_layout=True)

#     bar_width = 0.2
#     bars1 = plt.bar(index, values_1, width=bar_width, color='blue', label=f'3DPAWS TSMS0{station_map[i][0]}')
#     bars2 = plt.bar([i + bar_width for i in index], values_2, width=bar_width, color='orange', label=f'3DPAWS TSMS0{station_map[i][1]}')
#     bars3 = plt.bar([i + 2*bar_width for i in index], values_3, width=bar_width, color='green', label=f'3DPAWS TSMS0{station_map[i][2]}')
#     bars4 = plt.bar([i + 3*bar_width for i in index], values_tsms, width=bar_width, color='red', label='TSMS Reference')

#     plt.xlabel('Month', fontsize=8)
#     plt.ylabel('Monthly Rainfall (mm)', fontsize=8)
#     plt.title(f'{sites[i]} Monthly Rainfall Comparison: All Instruments', fontsize=10)
#     plt.xticks([i + 1.5*bar_width for i in index], months, rotation=45, ha='right')
#     plt.legend(fontsize='small')

#     for bars in [bars1, bars2, bars3, bars4]:
#         for bar in bars:
#             yval = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom', fontsize=6)

#     plt.savefig(data_destination+sites[i]+f"\\total_rainfall\\{sites[i]}_monthly_rainfall_all_instruments.png")
    
#     plt.clf()
#     plt.close()


"""
=============================================================================================================================
Create time series DIFFERENCE plots for pressure compared to TSMS at each site. COMPLETE RECORDS
Complete record with daily average difference
=============================================================================================================================
"""
# print("Pressure difference")

# sites = {
#     0:"Ankara", 1:"Konya", 2:"Adana"
# }
# station_map = {
#     0:[0,1,2], 1:[3,4,5], 2:[6,7,8]
# }
# sht_upgrade = [ # the day the upgrade from HTU to SHT was performed
#     "2024-01-17", "2024-01-12", "2024-01-15"
# ] # same day that radiation shield sensor array updated

# for i in range(3):
#     print(f"\t{sites[i]}: Pressure Difference per Sensor")

#     inst_1 = paws_dfs[station_map[i][0]].copy(deep=True)
#     inst_2 = paws_dfs[station_map[i][1]].copy(deep=True)
#     inst_3 = paws_dfs[station_map[i][2]].copy(deep=True)
#     tsms_ref = tsms_dfs[i*3].copy(deep=True)

#     inst_1.reset_index(inplace=True)
#     inst_2.reset_index(inplace=True)
#     inst_3.reset_index(inplace=True)
#     tsms_ref.reset_index(inplace=True)

#     # Step 1: Calculate the daily mean difference for each temperature sensor from the reference.
#     daily_mean_diffs = []
#     for year_month_day in set(inst_1['year_month_day']) & \
#                         set(inst_2['year_month_day']) & \
#                             set(inst_3['year_month_day']) & \
#                                 set(tsms_ref['year_month_day']):    # not ordered when working w/ sets

#         inst_1_grouped = inst_1[inst_1['year_month_day'] == year_month_day]
#         inst_2_grouped = inst_2[inst_2['year_month_day'] == year_month_day]
#         inst_3_grouped = inst_3[inst_3['year_month_day'] == year_month_day]
#         tsms_grouped = tsms_ref[tsms_ref['year_month_day'] == year_month_day]

#         merged_df = pd.merge(inst_1_grouped, inst_2_grouped, on='date', suffixes=('_1', '_2'))
#         merged_df = pd.merge(merged_df, inst_3_grouped, on='date', suffixes=('', '_3'))
#         merged_df = pd.merge(merged_df, tsms_grouped, on='date', suffixes=('', '_tsms'))

#         for sensor in variable_mapper['actual_pressure']:
#             merged_df[f"{sensor}_1_diff"] = merged_df[f'{sensor}_1'] - merged_df['actual_pressure'] # 3D-PAWS instrument 1
#             merged_df[f"{sensor}_2_diff"] = merged_df[f'{sensor}_2'] - merged_df['actual_pressure'] # 3D-PAWS instrument 2
#             merged_df[f"{sensor}_diff"] = merged_df[f'{sensor}'] - merged_df['actual_pressure']     # 3D-PAWS instrument 3

#             daily_mean_diffs.append({
#                 'date':year_month_day,
#                 'sensor':sensor,
#                 f'TSMS{station_map[i][0]}_diff':merged_df[f"{sensor}_1_diff"].mean(),
#                 f'TSMS{station_map[i][1]}_diff':merged_df[f"{sensor}_2_diff"].mean(),
#                 f'TSMS{station_map[i][2]}_diff':merged_df[f"{sensor}_diff"].mean()
#             })

#     # Step 2: Plot average daily diff's for each sensor.
#     df = pd.DataFrame(daily_mean_diffs)
#     df['date'] = pd.to_datetime(df['date'].astype(str))

#     for sensor in variable_mapper['actual_pressure']:
#         this_sensor = df[df['sensor'] == sensor].sort_values('date')

#         plt.figure(figsize=(20, 12))

#         plt.plot(this_sensor['date'], this_sensor[f'TSMS{station_map[i][0]}_diff'], marker='.', markersize=1, label=f"TSMS0{station_map[i][0]} {sensor}")
#         plt.plot(this_sensor['date'], this_sensor[f'TSMS{station_map[i][1]}_diff'], marker='.', markersize=1, label=f"TSMS0{station_map[i][1]} {sensor}")
#         plt.plot(this_sensor['date'], this_sensor[f'TSMS{station_map[i][2]}_diff'], marker='.', markersize=1, label=f"TSMS0{station_map[i][2]} {sensor}")

#         plt.title(f'{sites[i]}: {sensor} Pressure Difference from Reference')
#         plt.xlabel('Date')
#         plt.ylabel('Pressure Difference [Actual] (hPa)')
#         plt.ylim(-5, 5)
#         plt.axvline(pd.to_datetime(sht_upgrade[i]), color='red', linestyle='--', linewidth=2)
#         plt.xticks(rotation=45)

#         plt.legend()

#         plt.grid(True)
#         plt.tight_layout()

#         #plt.savefig(data_destination+station_directories[i]+f"total_rainfall/raw/{station_directories[i][8:14]}_rainfall_accumulation_TEST.png")
#         plt.savefig(data_destination+sites[i]+f"\\actual_pressure\\difference-plots\\{sites[i]}_actual_pressure_difference.png")

#         plt.clf()
#         plt.close()


"""
=============================================================================================================================
Create time series DIFFERENCE plots for each temperature sensor compared to TSMS at each site. COMPLETE RECORDS
Daily average difference
=============================================================================================================================
"""
# print("Temperature difference")

# sites = {
#     0:"Ankara", 1:"Konya", 2:"Adana"
# }
# station_map = {
#     0:[0,1,2], 1:[3,4,5], 2:[6,7,8]
# }
# sht_upgrade = [ # the day the upgrade from HTU to SHT was performed
#     "2024-01-17", "2024-01-12", "2024-01-15"
# ] # same day that radiation shield sensor array updated

# for i in range(3):
#     print(f"\t{sites[i]}: Temperature Difference per Sensor")

#     inst_1 = paws_dfs[station_map[i][0]].copy(deep=True)
#     inst_2 = paws_dfs[station_map[i][1]].copy(deep=True)
#     inst_3 = paws_dfs[station_map[i][2]].copy(deep=True)
#     tsms_ref = tsms_dfs[i*3].copy(deep=True)

#     inst_1.reset_index(inplace=True)
#     inst_2.reset_index(inplace=True)
#     inst_3.reset_index(inplace=True)
#     tsms_ref.reset_index(inplace=True)

#     # Step 1: Calculate the daily mean difference for each temperature sensor from the reference.
#     daily_mean_diffs = []
#     for year_month_day in set(inst_1['year_month_day']) & \
#                         set(inst_2['year_month_day']) & \
#                             set(inst_3['year_month_day']) & \
#                                 set(tsms_ref['year_month_day']):    # not ordered when working w/ sets

#         inst_1_grouped = inst_1[inst_1['year_month_day'] == year_month_day]
#         inst_2_grouped = inst_2[inst_2['year_month_day'] == year_month_day]
#         inst_3_grouped = inst_3[inst_3['year_month_day'] == year_month_day]
#         tsms_grouped = tsms_ref[tsms_ref['year_month_day'] == year_month_day]

#         merged_df = pd.merge(inst_1_grouped, inst_2_grouped, on='date', suffixes=('_1', '_2'))
#         merged_df = pd.merge(merged_df, inst_3_grouped, on='date', suffixes=('', '_3'))
#         merged_df = pd.merge(merged_df, tsms_grouped, on='date', suffixes=('', '_tsms'))

#         for sensor in variable_mapper['temperature']:
#             merged_df[f"{sensor}_1_diff"] = merged_df[f'{sensor}_1'] - merged_df['temperature'] # 3D-PAWS instrument 1
#             merged_df[f"{sensor}_2_diff"] = merged_df[f'{sensor}_2'] - merged_df['temperature'] # 3D-PAWS instrument 2
#             merged_df[f"{sensor}_diff"] = merged_df[f'{sensor}'] - merged_df['temperature']     # 3D-PAWS instrument 3

#             daily_mean_diffs.append({
#                 'date':year_month_day,
#                 'sensor':sensor,
#                 f'TSMS{station_map[i][0]}_diff':merged_df[f"{sensor}_1_diff"].mean(),
#                 f'TSMS{station_map[i][1]}_diff':merged_df[f"{sensor}_2_diff"].mean(),
#                 f'TSMS{station_map[i][2]}_diff':merged_df[f"{sensor}_diff"].mean()
#             })

#     # Step 2: Plot average daily diff's for each sensor.
#     df = pd.DataFrame(daily_mean_diffs)
#     df['date'] = pd.to_datetime(df['date'].astype(str))

#     for sensor in variable_mapper['temperature']:
#         this_sensor = df[df['sensor'] == sensor].sort_values('date')

#         plt.figure(figsize=(20, 12))

#         plt.plot(this_sensor['date'], this_sensor[f'TSMS{station_map[i][0]}_diff'], marker='.', markersize=1, label=f"TSMS0{station_map[i][0]} {sensor}")
#         plt.plot(this_sensor['date'], this_sensor[f'TSMS{station_map[i][1]}_diff'], marker='.', markersize=1, label=f"TSMS0{station_map[i][1]} {sensor}")
#         plt.plot(this_sensor['date'], this_sensor[f'TSMS{station_map[i][2]}_diff'], marker='.', markersize=1, label=f"TSMS0{station_map[i][2]} {sensor}")

#         plt.title(f'{sites[i]}: {sensor} Temperature Difference from Reference')
#         plt.xlabel('Date')
#         plt.ylabel('Temperature Difference (˚C)')
#         plt.ylim(-5, 5)
#         plt.axvline(pd.to_datetime(sht_upgrade[i]), color='red', linestyle='--', linewidth=2)
#         plt.xticks(rotation=45)

#         plt.legend()

#         plt.grid(True)
#         plt.tight_layout()

#         #plt.savefig(data_destination+station_directories[i]+f"total_rainfall/raw/{station_directories[i][8:14]}_rainfall_accumulation_TEST.png")
#         plt.savefig(data_destination+sites[i]+f"\\temperature\\difference-plots\\{sites[i]}_{sensor}_temperature_difference.png")

#         plt.clf()
#         plt.close()
            

"""
=============================================================================================================================
Create time series DIFFERENCE plots for each humidity sensor compared to TSMS at each site. MONTHLY RECORDS
Daily average difference
=============================================================================================================================
"""
# print("Humidity difference")

# sites = {
#     0:"Ankara", 1:"Konya", 2:"Adana"
# }
# station_map = {
#     0:[0,1,2], 1:[3,4,5], 2:[6,7,8]
# }
# sht_upgrade = [ # the day the upgrade from HTU to SHT was performed
#     "2024-01-17", "2024-01-12", "2024-01-15"
# ] # same day that radiation shield sensor array updated

# for i in range(3):
#     print(f"\t{sites[i]}: Humidity Difference per Sensor")

#     inst_1 = paws_dfs[station_map[i][0]].copy(deep=True)
#     inst_2 = paws_dfs[station_map[i][1]].copy(deep=True)
#     inst_3 = paws_dfs[station_map[i][2]].copy(deep=True)
#     tsms_ref = tsms_dfs[i*3].copy(deep=True)

#     inst_1.reset_index(inplace=True)
#     inst_2.reset_index(inplace=True)
#     inst_3.reset_index(inplace=True)
#     tsms_ref.reset_index(inplace=True)

#     # Step 1: Calculate the daily mean difference for each temperature sensor from the reference.
#     daily_mean_diffs = []
#     for year_month_day in set(inst_1['year_month_day']) & \
#                         set(inst_2['year_month_day']) & \
#                             set(inst_3['year_month_day']) & \
#                                 set(tsms_ref['year_month_day']):    # not ordered when working w/ sets

#         inst_1_grouped = inst_1[inst_1['year_month_day'] == year_month_day]
#         inst_2_grouped = inst_2[inst_2['year_month_day'] == year_month_day]
#         inst_3_grouped = inst_3[inst_3['year_month_day'] == year_month_day]
#         tsms_grouped = tsms_ref[tsms_ref['year_month_day'] == year_month_day]

#         merged_df = pd.merge(inst_1_grouped, inst_2_grouped, on='date', suffixes=('_1', '_2'))
#         merged_df = pd.merge(merged_df, inst_3_grouped, on='date', suffixes=('', '_3'))
#         merged_df = pd.merge(merged_df, tsms_grouped, on='date', suffixes=('', '_tsms'))

#         for sensor in variable_mapper['humidity']:
#             merged_df[f"{sensor}_1_diff"] = merged_df[f'{sensor}_1'] - merged_df['humidity'] # 3D-PAWS instrument 1
#             merged_df[f"{sensor}_2_diff"] = merged_df[f'{sensor}_2'] - merged_df['humidity'] # 3D-PAWS instrument 2
#             merged_df[f"{sensor}_diff"] = merged_df[f'{sensor}'] - merged_df['humidity']     # 3D-PAWS instrument 3

#             daily_mean_diffs.append({
#                 'date':year_month_day,
#                 'sensor':sensor,
#                 f'TSMS{station_map[i][0]}_diff':merged_df[f"{sensor}_1_diff"].mean(),
#                 f'TSMS{station_map[i][1]}_diff':merged_df[f"{sensor}_2_diff"].mean(),
#                 f'TSMS{station_map[i][2]}_diff':merged_df[f"{sensor}_diff"].mean()
#             })

#     # Step 2: Plot average daily diff's for each sensor.
#     df = pd.DataFrame(daily_mean_diffs)
#     df['date'] = pd.to_datetime(df['date'].astype(str))

#     for sensor in variable_mapper['humidity']:
#         if sensor == 'bme2_hum': continue

#         this_sensor = df[df['sensor'] == sensor].sort_values('date')

#         plt.figure(figsize=(20, 12))

#         plt.plot(this_sensor['date'], this_sensor[f'TSMS{station_map[i][0]}_diff'], marker='.', markersize=1, label=f"TSMS0{station_map[i][0]} {sensor}")
#         plt.plot(this_sensor['date'], this_sensor[f'TSMS{station_map[i][1]}_diff'], marker='.', markersize=1, label=f"TSMS0{station_map[i][1]} {sensor}")
#         plt.plot(this_sensor['date'], this_sensor[f'TSMS{station_map[i][2]}_diff'], marker='.', markersize=1, label=f"TSMS0{station_map[i][2]} {sensor}")

#         plt.title(f'{sites[i]}: {sensor} Humidity Difference from Reference')
#         plt.xlabel('Date')
#         plt.ylabel('Humidity Difference (%)')
#         plt.ylim(-50, 50)
#         plt.axvline(pd.to_datetime(sht_upgrade[i]), color='red', linestyle='--', linewidth=2)
#         plt.xticks(rotation=45)

#         plt.legend()

#         plt.grid(True)
#         plt.tight_layout()

#         #plt.savefig(data_destination+station_directories[i]+f"total_rainfall/raw/{station_directories[i][8:14]}_rainfall_accumulation_TEST.png")
#         plt.savefig(data_destination+sites[i]+f"\\humidity\\difference-plots\\{sites[i]}_{sensor}_humidity_difference.png")

#         plt.clf()
#         plt.close()
