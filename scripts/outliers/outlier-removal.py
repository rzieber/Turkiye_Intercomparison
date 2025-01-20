"""
HTU filtering methodology, and other statistical outlier removal tactics. 

Requires station_TSMS00 -> 08 folder struct, with the complete records for TSMS and 3D-PAWS within each folder
within the data_origin directory.
The data_destination pathway doesn't require this, just list the full pathname where you want things stored.

Phase 1: Nulls (-999.99)
Phase 2: Timestamp resets (out of order timestamps)
Phase 3: Thresholds (unrealistic values)
Phase 4: Manual removal of identified special cases
Phase 5: HTU filtering
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from windrose import WindroseAxes
import matplotlib.cm as cm
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import seaborn as sns
from pathlib import Path
from datetime import timedelta
from functools import reduce

sys.path.append(str(Path(__file__).resolve().parents[2]))
from resources import functions as func


data_origin = r"C:\\Users\\Becky\\Documents\\UCAR_ImportantStuff\\Turkiye\\data\\"
data_destination = r"C:\\Users\\Becky\\Documents\\UCAR_ImportantStuff\\Turkiye\\plots\\time_series\\rainfall_accumulation_per_site\\"

outlier_reasons = [
    "null", "timestamp reset", "threshold", "manual removal"
]

station_variables = [ 
    "temperature", "humidity", "actual_pressure", "sea_level_pressure", "wind", "total_rainfall"
]
variable_mapper = { # TSMS : 3DPAWS
    "temperature":["bmp2_temp", "htu_temp", "mcp9808"],
    "humidity":["bme2_hum", "htu_hum"],
    "actual_pressure":["bmp2_pres"],
    "sea_level_pressure":["bmp2_slp"],
    "avg_wind_dir":["wind_dir"], # both using 1-minute average
    "avg_wind_speed":["wind_speed"], # both usng 1-minute average
    "total_rainfall":["tipping"]
}
station_directories = [
    "station_TSMS00/", "station_TSMS01/", "station_TSMS02/",   
    "station_TSMS03/", "station_TSMS04/", "station_TSMS05/",
    "station_TSMS06/", "station_TSMS07/", "station_TSMS08/"
]

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
                    os.makedirs(data_destination+station_directories[i]+var+'/raw/', exist_ok=True)
            except Exception as e:
                print("Error creating the directory: ", var)
    except Exception as e:
        print("Could not create ", station_directories[i][:len(station_directories[i])-1])
    
# -------------------------------------------------------------------------------------------------------------------------------
paws_dfs = []
tsms_dfs = []

for i in range(len(station_directories)):
    paws_df = tsms_df = None
    """
    =============================================================================================================================
    Create the 3D PAWS and TSMS dataframes.
    =============================================================================================================================
    """
    for file in station_files[i]:
        if file.startswith("TSMS"): # 3D-PAWS records start w/ 'TSMS##', TSMS ref. records start w/ the site name
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

    if station_directories[i] == "station_TSMS06/": paws_df_FILTERED.to_csv(f"C:\\Users\\Becky\\Downloads\\tsms06_pre-removal.csv")
    """
    ============================================================================================================================
    Phase 4: Manual removal of known outliers.
    ============================================================================================================================
    """
    print(f"Phase 4: Filtering out known outliers (special cases -- see Phase 4 for notes).") 

    # Erroneously high values -- most likely test tips during fabrication process
    if station_directories[i] == "station_TSMS04/":
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
    if station_directories[i] == "station_TSMS08/":
        existing_nulls = paws_df_FILTERED['tipping'].isnull() 

        dates_to_remove = [
            ("2023-04-08","2023-04-14 23:59:00"),           # Possibly faulty connector
            ("2023-05-15 00:00:00","2023-05-17 23:59:59"),  # Part I TBD
            ("2024-02-08 00:00:00","2024-02-11 23:59:59"),  # Part II TBD
            ("2024-02-15 00:00:00","2024-02-20 23:59:59"), 
            ("2024-03-07 00:00:00","2024-03-07 23:59:59"),
            ("2024-03-09 00:00:00","2024-03-10 23:59:59"), 
            ("2024-03-21 00:00:00","2024-03-21 23:59:59"),
            #("2024-09-21 00:00:00","2024-09-21 23:59:59")   # Part III TBD (is it sensor malfunction or something to keep?)
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
            'original_value': all_removed.values,
            'outlier_type': outlier_reasons[3]
        })
        paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)

    # """
    # =============================================================================================================================
    # Phase 5: Filter out HTU bit-switching. 3D-PAWS only.
    # =============================================================================================================================
    # """
    # print(f"Phase 5: Filtering out HTU trend-switching.")

    # paws_df_FILTERED.reset_index(drop=True, inplace=True)   # Reset the index to ensure it is a simple range

    # # isolate the known timeframes when HTU bit-switching is happening
    # # create a list of verified starting & ending timestamps to iterate through

    # for variable in variable_mapper:                                
    #     if variable in ["actual_pressure", "sea_level_pressure", "avg_wind_speed", "avg_wind_dir", "total_rainfall"]: continue

    #     existing_nulls = paws_df_FILTERED[variable].isnull()   

    #     tsms_df_FILTERED[variable] = pd.to_numeric(tsms_df_FILTERED[variable], errors='coerce')


    # """
    # =============================================================================================================================
    # Phase 6: Filter out contextual outliers using statistical methods
    # =============================================================================================================================
    # """
    # print(f"Phase 6: Filtering out contextual outliers using statistical methods.")

    # tsms_df_FILTERED.reset_index(drop=True, inplace=True)           # Reset the index to ensure it is a simple range
    # paws_df_FILTERED.reset_index(drop=True, inplace=True)

    # for variable in variable_mapper:                                # Filtering TSMS --------------------------------------------
    #     if variable in ["avg_wind_speed", "avg_wind_dir", "total_rainfall"]: continue

    #     existing_nulls = tsms_df_FILTERED[variable].isnull()   

    #     tsms_df_FILTERED[variable] = pd.to_numeric(tsms_df_FILTERED[variable], errors='coerce')

    #     # IQR -------------------
    #     print("\t\tIQR")
    #     Q1 = tsms_df_FILTERED[variable].quantile(0.25)
    #     Q3 = tsms_df_FILTERED[variable].quantile(0.75)
        
    #     IQR = Q3 - Q1
        
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
        
    #     mask_IQR = (tsms_df_FILTERED[variable] < lower_bound) | (tsms_df_FILTERED[variable] > upper_bound)

    #     new_null_IQR = mask_IQR &  ~existing_nulls 
    #     tsms_df_FILTERED.loc[new_null_IQR, variable] = np.nan
    #     # -----------------------

    #     # z-score ---------------
    #     print("\t\tZ-score")
    #     window = 20                                               
    #     tsms_df_FILTERED['rolling_mean'] = tsms_df_FILTERED[variable].rolling(window=window).mean() 
    #     tsms_df_FILTERED['rolling_std'] = tsms_df_FILTERED[variable].rolling(window=window).std()

    #     tsms_df_FILTERED['z-score'] = (tsms_df_FILTERED[variable] - tsms_df_FILTERED['rolling_mean']) / tsms_df_FILTERED['rolling_std']         

    #     threshold = 3                                              

    #     mask_Zscore = tsms_df_FILTERED['z-score'].abs() > threshold   
    #     new_null_Zscore = mask_Zscore & ~existing_nulls
    #     tsms_df_FILTERED.loc[new_null_Zscore, variable] = np.nan 

    #     tsms_df_FILTERED.drop(columns=['rolling_mean', 'rolling_std', 'z-score'], inplace=True)    
    #     # -----------------------


    #     # # Abs Diff --------------         TSMS DOESN'T NEED HTU FILTERING BECAUSE    D U H !
    #     # print("\t\tAbsolute Diff")

    #     # threshold = None
    #     # last_non_null = None
    #     # diff_outliers = []

    #     # i = 0
    #     # while i < len(tsms_df_FILTERED) - 2:
    #     #     if pd.isna(last_non_null): print("Error with ", last_non_null, ". Current index: ", i)
    #     #     if pd.notna(tsms_df_FILTERED.loc[i, variable]): 
    #     #         last_non_null = tsms_df_FILTERED.loc[i, variable]
    #     #     if pd.isna(tsms_df_FILTERED.loc[i+1, variable]): 
    #     #         i += 1
    #     #         continue

    #     #     if pd.isna(last_non_null) and i == 0: continue

    #     #     if variable == "temp": threshold = 2
    #     #     elif variable == "humidity": threshold = 2
    #     #     elif variable in ["actual_pressure", "sea_level_pressure"]: threshold = 2

    #     #     diff = abs(tsms_df_FILTERED.loc[i+1, variable] - last_non_null)

    #     #     if diff is None or last_non_null is None: 
    #     #             print("Diff: ", diff, "\t\tType: ", type(diff))
    #     #             print("Last non null: ", last_non_null, "\t\tType: ", type(last_non_null))
    #     #             print("Index: ", i)
    #     #             continue

    #     #     if diff > threshold: 
    #     #         diff_outliers.append({
    #     #             'date': tsms_df_FILTERED.loc[i+1, 'date'],
    #     #             'column_name': variable,
    #     #             'original_value': tsms_df_FILTERED.loc[i+1, variable],
    #     #             'outlier_type': 'diff contextual'
    #     #         })

    #     #         tsms_df_FILTERED.loc[i+1, variable] = np.nan

    #     #     i += 1
        
    #     # diff_outliers_df = pd.DataFrame(diff_outliers)
    #     # # -----------------------         
        
    #     outliers_to_add = pd.DataFrame({
    #         'date': tsms_df_FILTERED.loc[new_null_IQR, 'date'],
    #         'column_name': variable,
    #         'original_value': tsms_df_FILTERED.loc[new_null_IQR, variable],
    #         'outlier_type': 'IQR contextual'
    #     })
    #     tsms_outliers = pd.concat([tsms_outliers, outliers_to_add], ignore_index=True)
        
    #     outliers_to_add = pd.DataFrame({
    #         'date': tsms_df_FILTERED.loc[new_null_Zscore, 'date'],
    #         'column_name': variable,
    #         'original_value': tsms_df_FILTERED.loc[new_null_Zscore, variable],
    #         'outlier_type': 'z-score contextual'
    #     })
    #     tsms_outliers = pd.concat([tsms_outliers, outliers_to_add], ignore_index=True)

    #     tsms_outliers = pd.concat([tsms_outliers, diff_outliers_df], ignore_index=True)


    #     outliers_to_add = None


    #     for var in variable_mapper[variable]:               # Filtering PAWS ----------------------------------------------------
    #         existing_nulls = paws_df_FILTERED[var].isnull()

    #         paws_df_FILTERED[var] = pd.to_numeric(paws_df_FILTERED[var], errors='coerce')

    #         # Abs Diff -------------- FOR THE HTU ONLY!!!!!!!!
    #         if var != "htu_hum": continue;

    #         threshold = 1.2
    #         last_non_null = None
    #         diff_outliers = []

    #         i = 0
    #         while i < len(paws_df_FILTERED) - 2:
    #             if pd.isna(last_non_null): print("Error with ", last_non_null, ". Current index: ", i)
                
    #             if pd.notna(paws_df_FILTERED.loc[i, var]): 
    #                 last_non_null = paws_df_FILTERED.loc[i, var]
    #             if pd.isna(paws_df_FILTERED.loc[i+1, var]): 
    #                 i += 1
    #                 continue

    #             if pd.isna(last_non_null) and i == 0: continue

    #             diff = abs(paws_df_FILTERED.loc[i+1, var] - last_non_null)
                
    #             if diff is None or last_non_null is None: 
    #                 print("Diff: ", diff, "\t\tType: ", type(diff))
    #                 print("Last non null: ", last_non_null, "\t\tType: ", type(last_non_null))
    #                 print(i)
    #                 i += 1
    #                 continue

    #             if diff > threshold: 
    #                 diff_outliers.append({
    #                     'date': paws_df_FILTERED.loc[i+1, 'date'],
    #                     'column_name': var,
    #                     'original_value': paws_df_FILTERED.loc[i+1, var],
    #                     'outlier_type': 'diff contextual'
    #                 })

    #                 paws_df_FILTERED.loc[i+1, var] = np.nan

    #             i += 1
            
    #         diff_outliers_df = pd.DataFrame(diff_outliers)
    #         # -----------------------   

    #         # # IQR -------------------
    #         # Q1 = paws_df_FILTERED[var].quantile(0.25)
    #         # Q3 = paws_df_FILTERED[var].quantile(0.75)
            
    #         # IQR = Q3 - Q1
            
    #         # lower_bound = Q1 - 1.5 * IQR
    #         # upper_bound = Q3 + 1.5 * IQR
            
    #         # mask_IQR = (paws_df_FILTERED[var] < lower_bound) | (paws_df_FILTERED[var] > upper_bound)

    #         # new_null_IQR = mask_IQR & ~existing_nulls 
    #         # paws_df_FILTERED.loc[new_null_IQR, var] = np.nan 
    #         # # -----------------------

    #         # # z-score ---------------
    #         # paws_df_FILTERED['rolling_mean'] = paws_df_FILTERED[var].rolling(window=window).mean()                   
    #         # paws_df_FILTERED['rolling_std'] = paws_df_FILTERED[var].rolling(window=window).std()

    #         # paws_df_FILTERED['z-score'] = (paws_df_FILTERED[var] - paws_df_FILTERED['rolling_mean']) / paws_df_FILTERED['rolling_std']           

    #         # threshold = 3                                      

    #         # mask_Zscore = paws_df_FILTERED['z-score'].abs() > threshold  
    #         # new_null_Zscore = mask_Zscore & ~existing_nulls
    #         # paws_df_FILTERED.loc[new_null_Zscore, var] = np.nan 

    #         # paws_df_FILTERED.drop(columns=['rolling_mean', 'rolling_std', 'z-score'], inplace=True)  
    #         # # -----------------------

    #         outliers_to_add = pd.DataFrame({
    #             'date': paws_df_FILTERED.loc[new_null_IQR, 'date'],
    #             'column_name': var,
    #             'original_value': paws_df_FILTERED.loc[new_null_IQR, var],
    #             'outlier_type': 'IQR contextual'
    #         })
    #         paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)

    #         outliers_to_add = pd.DataFrame({
    #             'date': paws_df_FILTERED.loc[new_null_Zscore, 'date'],
    #             'column_name': var,
    #             'original_value': paws_df_FILTERED.loc[new_null_Zscore, var],
    #             'outlier_type': 'z-score contextual'
    #         })
    #         paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)

    #         paws_outliers = pd.concat([paws_outliers, diff_outliers_df], ignore_index=True)


    paws_dfs.append(paws_df_FILTERED)
    tsms_dfs.append(tsms_df_FILTERED)

    print("\n----------------------")

# ----------------------------------------------------------------------------------------------------------------------------------------------- PLOT GEN LOGIC
    
    """
    =============================================================================================================================
    Create a time series plot of the 3D PAWS station data versus the TSMS reference station. MONTHLY RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: " \
    #                    "Time-series plots for temperature, humidity, actual pressure and sea level pressure -- monthly records")
    
    # paws_df_FILTERED.set_index('date', inplace=True)
    # tsms_df_FILTERED.set_index('date', inplace=True)
    
    # tsms_df_FILTERED = tsms_df_FILTERED[paws_df_FILTERED.index[0]:paws_df_FILTERED.index[-1]]

    # # Temperature ---------------------------------------------------------------------------------------------------------------
    # print("Temperature -- raw")
    # for t in variable_mapper['temperature']:
    #     for year_month, paws_group in paws_df_FILTERED.groupby('year_month'):
    #         tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

    #         plt.figure(figsize=(20, 12))

    #         plt.plot(paws_group.index, paws_group[f'{t}'], marker='.', markersize=1, label=f"3D PAWS {t}")
    #         plt.plot(tsms_group.index, tsms_group['temperature'], marker='.', markersize=1, label='TSMS')
        
    #         plt.title(f'{station_directories[i][8:14]} Temperature for {year_month}: 3D PAWS {t} versus TSMS')
    #         plt.xlabel('Date')
    #         plt.ylabel('Temperature (˚C)')
    #         plt.xticks(rotation=45)

    #         plt.legend()

    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.savefig(data_destination+station_directories[i]+f"temp/raw/{station_directories[i][8:14]}_{t}_{year_month}.png")
            
    #         plt.clf()
    #         plt.close()


    # # Humidity ------------------------------------------------------------------------------------------------------------------
    # print("Humidity -- raw")
    # for h in variable_mapper['humidity']:
    #     for year_month, paws_group in paws_df_FILTERED.groupby('year_month'):
    #         tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

    #         plt.figure(figsize=(20, 12))

    #         plt.plot(paws_group.index, paws_group[f'{h}'], marker='.', markersize=1, label=f"3D PAWS {h}")
    #         plt.plot(tsms_group.index, tsms_group['humidity'], marker='.', markersize=1, label='TSMS')
        
    #         plt.title(f'{station_directories[i][8:14]} Humidity for {year_month}: 3D PAWS {h} versus TSMS')
    #         plt.xlabel('Date')
    #         plt.ylabel('Humidity (%)')
    #         plt.xticks(rotation=45)

    #         plt.legend()

    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.savefig(data_destination+station_directories[i]+f"humidity/raw/{station_directories[i][8:14]}_{h}_{year_month}.png")
            
    #         plt.clf()
    #         plt.close()


    # # Pressure ------------------------------------------------------------------------------------------------------------------
    # print("Pressure -- raw")
    # for year_month, paws_group in paws_df_FILTERED.groupby("year_month"):
    #     tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

    #     plt.figure(figsize=(20, 12))

    #     plt.plot(paws_group.index, paws_group['bmp2_pres'], marker='.', markersize=1, label="3D PAWS bmp2_pres")
    #     plt.plot(tsms_group.index, tsms_group['actual_pressure'], marker='.', markersize=1, label='TSMS')

    #     plt.title(f'{station_directories[i][8:14]} Pressure for {year_month}: 3D PAWS bmp2_pres versus TSMS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Pressure (mbar)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(data_destination+station_directories[i]+f"actual_pressure/raw/{station_directories[i][8:14]}_bmp2_pres_{year_month}.png")
        
    #     plt.clf()
    #     plt.close()

    
    # # Sea Level Pressure --------------------------------------------------------------------------------------------------------
    # print("Sea level pressure -- raw")
    # for year_month, paws_group in paws_df_FILTERED.groupby("year_month"):
    #     tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

    #     plt.figure(figsize=(20, 12))

    #     plt.plot(paws_group.index, paws_group['bmp2_slp'], marker='.', markersize=1, label="3D PAWS bmp2_slp")
    #     plt.plot(tsms_group.index, tsms_group['sea_level_pressure'], marker='.', markersize=1, label='TSMS')

    #     plt.title(f'{station_directories[i][8:14]} Sea Level Pressure for {year_month}: 3D PAWS bmp2_slp versus TSMS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Pressure (mbar)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(data_destination+station_directories[i]+f"sea_level_pressure/raw/{station_directories[i][8:14]}_bmp2_slp_{year_month}.png")
        
    #     plt.clf()
    #     plt.close()


"""
=============================================================================================================================
Create a time series plot of each individual 3D PAWS station data versus the TSMS reference station. COMPLETE RECORDS
=============================================================================================================================
"""
sites = {
    0:"Ankara", 1:"Konya", 2:"Adana"
}
station_map = {
    0:[0,1,2], 1:[3,4,5], 2:[6,7,8]
}

for i in range(3):
    print(f"\t{sites[i]}: Rainfall accumulation")

    inst_1 = paws_dfs[station_map[i][0]].copy(deep=True)
    inst_2 = paws_dfs[station_map[i][1]].copy(deep=True)
    inst_3 = paws_dfs[station_map[i][2]].copy(deep=True)
    tsms_ref = tsms_dfs[i].copy(deep=True)

    inst_1.reset_index(inplace=True)
    inst_2.reset_index(inplace=True)
    inst_3.reset_index(inplace=True)
    tsms_ref.reset_index(inplace=True)

    merged_df = pd.merge(inst_1, inst_2, on='date', suffixes=('_1', '_2'))
    merged_df = pd.merge(merged_df, inst_3, on='date', suffixes=('', '_3'))
    merged_df = pd.merge(merged_df, tsms_ref, on='date', suffixes=('', '_tsms'))

    merged_df['cumulative_rainfall_3DPAWS_1'] = merged_df['tipping_1'].cumsum()
    merged_df['cumulative_rainfall_3DPAWS_2'] = merged_df['tipping_2'].cumsum()
    merged_df['cumulative_rainfall_3DPAWS_3'] = merged_df['tipping'].cumsum()
    merged_df['cumulative_rainfall_TSMS'] = merged_df['total_rainfall'].cumsum()

    plt.figure(figsize=(20, 12))

    plt.plot(merged_df['date'], merged_df['cumulative_rainfall_3DPAWS_1'], marker='.', markersize=1, label=f"TSMS0{station_map[i][0]} 3D PAWS")
    plt.plot(merged_df['date'], merged_df['cumulative_rainfall_3DPAWS_2'], marker='.', markersize=1, label=f"TSMS0{station_map[i][1]} 3D PAWS")
    plt.plot(merged_df['date'], merged_df['cumulative_rainfall_3DPAWS_3'], marker='.', markersize=1, label=f"TSMS0{station_map[i][2]} 3D PAWS")
    plt.plot(merged_df['date'], merged_df['cumulative_rainfall_TSMS'], marker='.', markersize=1, label=f'{sites[i]} TSMS Reference')

    plt.title(f'{sites[i]} Rainfall Accumulation')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    plt.xticks(rotation=45)

    plt.legend()

    plt.grid(True)
    plt.tight_layout()

    #plt.savefig(data_destination+station_directories[i]+f"total_rainfall/raw/{station_directories[i][8:14]}_rainfall_accumulation_TEST.png")
    plt.savefig(data_destination+sites[i]+f"\\total_rainfall\\{sites[i]}_rainfall_accumulation.png")

    plt.clf()
    plt.close()


    """
    =============================================================================================================================
    Create a time series plot of each 3D PAWS station data versus the TSMS reference station. COMPLETE RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: Cumulative rainfall -- complete records [PER INSTRUMENT]")

    # for paws_station in station_directories: 
    #     merged_df = pd.merge(paws_df_FILTERED, tsms_df_FILTERED, on='date', how='inner')  # to eliminate bias

    #     merged_df['cumulative_rainfall_3DPAWS'] = merged_df['tipping'].cumsum()
    #     merged_df['cumulative_rainfall_TSMS'] = merged_df['total_rainfall'].cumsum()

    #     plt.figure(figsize=(20, 12))

    #     plt.plot(merged_df['date'], merged_df['cumulative_rainfall_3DPAWS'], marker='.', markersize=1, label="3D PAWS")
    #     plt.plot(merged_df['date'], merged_df['cumulative_rainfall_TSMS'], marker='.', markersize=1, label='TSMS')

    #     plt.title(f'{station_directories[i][8:14]} Rainfall Accumulation: 3D PAWS versus TSMS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Rainfall (mm)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()

    #     #plt.savefig(data_destination+station_directories[i]+f"total_rainfall/raw/{station_directories[i][8:14]}_rainfall_accumulation_TEST.png")
    #     plt.savefig(data_destination+station_directories[i]+f"total_rainfall\\raw\\{station_directories[i][8:14]}_rainfall_accumulation.png")

    #     plt.clf()
    #     plt.close()


    """
    =============================================================================================================================
    Create bar charts for monthly 3D PAWS rainfall accumulation compared to TSMS rainfall accumulation. COMPLETE RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: " \
    #                    "Bar charts for rainfall accumulation -- complete records")
    
    # paws_df_FILTERED_2 = paws_df_FILTERED[paws_df_FILTERED['tipping'] >= 0].copy().reset_index()
    # tsms_df_FILTERED_2 = tsms_df_FILTERED[(tsms_df_FILTERED['total_rainfall']) >= 0].copy().reset_index()

    # if station_directories[i] == "station_TSMS04/":
    #     exclude_garbage = pd.to_datetime("2022-09-01")
    #     tsms_df_REDUCED = tsms_df_FILTERED_2[
    #         (tsms_df_FILTERED_2['date'] >= exclude_garbage) & \
    #         (tsms_df_FILTERED_2['date'] <= paws_df_FILTERED_2['date'].iloc[-1])
    #     ]
    #     paws_df_FILTERED_2 = paws_df_FILTERED_2[
    #         (paws_df_FILTERED_2['date'] >= exclude_garbage) & \
    #         (paws_df_FILTERED_2['date'] <= paws_df_FILTERED_2['date'].iloc[-1])
    #     ]
    # elif station_directories[i] == "station_TSMS08/":
    #     exclude_garbage_start = pd.to_datetime("2023-04-08")
    #     exclude_garbage_end = pd.to_datetime("2023-04-15")
    #     paws_df_FILTERED_2 = paws_df_FILTERED_2[
    #         ~((paws_df_FILTERED_2['date'] >= exclude_garbage_start) & (paws_df_FILTERED_2['date'] <= exclude_garbage_end))
    #     ]
    #     tsms_df_REDUCED = tsms_df_FILTERED_2
    # else:
    #     tsms_df_REDUCED = tsms_df_FILTERED_2[ # need to filter TSMS s.t. data timeframe equals PAWS
    #         (tsms_df_FILTERED_2['date'] >= paws_df_FILTERED_2['date'].iloc[0]) & \
    #         (tsms_df_FILTERED_2['date'] <= paws_df_FILTERED_2['date'].iloc[-1])
    #     ]

    # paws_totals = {}
    # tsms_totals = {}

    # for year_month, paws_grouped in paws_df_FILTERED_2.groupby("year_month"):
    #     tsms_grouped = tsms_df_REDUCED[tsms_df_REDUCED['year_month'] == year_month]

    #     print(f"{station_directories[i][8:14]} at {year_month}")

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

    # plt.figure(figsize=(20, 12))

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

    # # Display the plot
    # plt.tight_layout()

    # # Save the plot
    # plt.savefig(data_destination+station_directories[i]+f"total_rainfall/raw/{station_directories[i][8:14]}_monthly_rainfall.png")    
    
    # plt.clf()
    # plt.close()   

    """
    =============================================================================================================================
    Create bar charts for daily 3D PAWS rainfall accumulation (per instrument) compared to TSMS rainfall accumulation. MONTHLY RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: " \
    #                     "Bar charts for rainfall accumulation -- monthly records [INSTRUMENT VS REFERENCE COMPARISON]")
        
    # for year_month, paws_grouped in paws_df_FILTERED.reset_index(inplace=True).groupby("year_month"):
    #     tsms_grouped = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

    #     print(f"{station_directories[i][8:14]} at {year_month}")

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
    #     plt.savefig(data_destination + station_directories[i] + f"total_rainfall\\raw\\{station_directories[i][8:14]}_{year_month}_daily_rainfall.png")    
        
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
#     tsms_ref = tsms_dfs[i].copy(deep=True)

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

#         print(f"{sites[i]} at {year_month}")

#         merged_df = pd.merge(inst_1_grouped, inst_2_grouped, on='date', suffixes=('_1', '_2'))
#         merged_df = pd.merge(merged_df, inst_3_grouped, on='date', suffixes=('', '_3'))
#         merged_df = pd.merge(merged_df, tsms_grouped, on='date', suffixes=('', '_tsms'))

#         merged_df['daily_rainfall_1'] = merged_df['tipping_1']    # Calculate daily rainfall totals
#         merged_df['daily_rainfall_2'] = merged_df['tipping_2']
#         merged_df['daily_rainfall_3'] = merged_df['tipping']
#         merged_df['daily_rainfall_TSMS'] = merged_df['total_rainfall']

#         #merged_df.to_csv(f"C:\\Users\\Becky\\Downloads\\{sites[i]}_{year_month}_merged_df_FINAL.csv")
#         #merged_df.to_csv(f"/Users/rzieber/Downloads/{sites[i]}_{year_month}_merged_df_FINAL.csv")
        
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

#         plt.savefig(data_destination + sites[i] + f"\\total_rainfall\\test\\{sites[i]}_{year_month}_daily_rainfall_all_instruments.png")    
        
#         plt.clf()
#         plt.close()

# ====================================================================================================    
# ====================================================================================================    
# ====================================================================================================    
    
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
#     tsms_ref = tsms_dfs[i].copy(deep=True)

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

#     #merged_df.to_csv(f"C:\\Users\\Becky\\Downloads\\{sites[i]}_merged_df_FINAL_1-19-25.csv")
#     #merged_df.to_csv(f"/Users/rzieber/Downloads/{sites[i]}_{year_month}_merged_df_FINAL.csv")

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

#     plt.savefig(data_destination + sites[i] + f"\\total_rainfall\\test\\{sites[i]}_monthly_rainfall_all_instruments.png")
    
#     plt.clf()
#     plt.close()

    # ====================================================================================================    
    # ====================================================================================================    
    # ====================================================================================================    





    # """
    # =============================================================================================================================
    # Create wind rose plots of the 3D PAWS station data as well as the TSMS reference station. COMPLETE RECORDS
    # =============================================================================================================================
    # """
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
    # labels = ['2.0-4.0 m/s', '4.0-6.0 m/s', '6.0-8.0 m/s', '8.0-10.0 m/s']

    # first_timestamp = pd.to_datetime(paws_df_FILTERED_2.index[0])
    # last_timestamp = pd.to_datetime(paws_df_FILTERED_2.index[-1])
    # # first_timestamp = pd.Timestamp("2022-09-09 00:00") # for reproducing TSMS study period
    # # last_timestamp = pd.Timestamp("2023-02-24 12:42")

    # paws_df_FILTERED_2.loc[:, 'wind_speed_category'] = pd.cut(paws_df_FILTERED_2['wind_speed'], bins=wind_speed_bins, labels=labels, right=False)

    # ax = WindroseAxes.from_ax()
    # ax.bar(paws_df_FILTERED_2['wind_dir'], paws_df_FILTERED_2['wind_speed'], normed=True, opening=0.8, edgecolor='white', bins=wind_speed_bins)
    # ax.set_legend(title=f"{station_directories[i][8:len(station_directories[i])-1]} 3D-PAWS Wind Rose")

    # # ax.set_rmax(35)
    # # ax.set_yticks([7, 14, 21, 28, 35])
    # # ax.set_yticklabels(['7%', '14%', '21%', '28%', '35%']) # for variable winds
    # ax.set_rmax(80)
    # ax.set_yticks([16, 32, 48, 64, 80])
    # ax.set_yticklabels(['16%', '32%', '48%', '64%', '80%']) # for NON-VARIABLE winds
        
    # ax.grid(True, linewidth=0.5)  # Thinner grid lines can improve readability

    # plt.savefig(data_destination+station_directories[i]+f"wind/raw/{station_directories[i][8:14]}_nonvariable_winds.png")
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
    
    # if i in [0,1,2]: ax.set_legend(title=f"Ankara TSMS Wind Rose")
    # elif i in [3,4,5]: ax.set_legend(title=f"Konya TSMS Wind Rose")
    # else: ax.set_legend(title=f"Adana TSMS Wind Rose")
    
    
    # # ax.set_rmax(35)
    # # ax.set_yticks([7, 14, 21, 28, 35])
    # # ax.set_yticklabels(['7%', '14%', '21%', '28%', '35%']) # for complete records
    # ax.set_rmax(80)
    # ax.set_yticks([16, 32, 48, 64, 80])
    # ax.set_yticklabels(['16%', '32%', '48%', '64%', '80%']) # for non-variable winds

    # ax.grid(True, linewidth=0.5)  # Thinner grid lines can improve readability

    # if i in [0,1,2]: plt.savefig(data_destination+station_directories[i]+f"wind/raw/Ankara_nonvariable_winds.png")
    # elif i in [3,4,5]: plt.savefig(data_destination+station_directories[i]+f"wind/raw/Konya_nonvariable_winds.png")
    # else: plt.savefig(data_destination+station_directories[i]+f"wind/raw/Adana_nonvariable_winds.png")
    
    # plt.clf()
    # plt.close()


    # """
    # =============================================================================================================================
    # Create time series plots of the 3 temperature sensors compared to TSMS. MONTHLY RECORDS
    # =============================================================================================================================
    # """
    # print(f"{station_directories[i][:len(station_directories[i])-1]} Temperature Comparison")

    # tsms_df_FILTERED_2 = tsms_df_FILTERED[paws_df_FILTERED.index[0]:paws_df_FILTERED.index[-1]]

    # tsms_df_FILTERED_2.reset_index(inplace=True)
    # paws_df_FILTERED.reset_index(inplace=True)

    # tsms_df_FILTERED_2['date'] = pd.to_datetime(tsms_df_FILTERED_2['date'])
    # paws_df_FILTERED['date'] = pd.to_datetime(paws_df_FILTERED['date'])

    # for year_month, paws_group in paws_df_FILTERED.groupby('year_month'):
    #     tsms_group = tsms_df_FILTERED_2[tsms_df_FILTERED_2['year_month'] == year_month]

    #     plt.figure(figsize=(20, 12))

    #     plt.plot(paws_group['date'], paws_group["bmp2_temp"], marker='.', markersize=1, label=f"3D PAWS bmp2_temp")
    #     plt.plot(paws_group['date'], paws_group["htu_temp"], marker='.', markersize=1, label=f"3D PAWS htu_temp")
    #     plt.plot(paws_group['date'], paws_group["mcp9808"], marker='.', markersize=1, label=f"3D PAWS mcp9808")
    #     plt.plot(tsms_group['date'], tsms_group['temperature'], marker='.', markersize=1, label='TSMS')

    #     plt.title(f'{station_directories[i][8:14]} Temperature Comparison for {year_month}: 3D PAWS versus TSMS')
    #     #plt.title(f'{station_directories[i][8:14]} Temperature Comparison for {year_month}: 3D PAWS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Temperature (˚C)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(data_destination+station_directories[i]+f"temperature/raw/{station_directories[i][8:14]}_temp_comparison_{year_month}.png")

    #     plt.clf()
    #     plt.close()

