            
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
from datetime import timedelta, datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from resources import functions

data_origin = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/output/"
data_destination = os.path.join(project_root, "plots/")
print(data_destination)

station_variables = [ 
    "temp", "humidity", "actual_pressure", "sea_level_pressure", "wind", "total_rainfall"
]
variable_mapper = { # TSMS : 3DPAWS
    "temp":["bmp2_temp", "htu_temp", "mcp9808"],
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
    """
    =============================================================================================================================
    Create the 3D PAWS and TSMS dataframes.
    =============================================================================================================================
    """
    paws_df = None 
    tsms_df = None
    for file in station_files[i]:
        if file.startswith("TSMS"): # the 3D PAWS dataset starts with 'TSMS' and the TSMS dataset starts with 'paws'  :')
            paws_df = pd.read_csv( 
                            data_origin+station_directories[i]+file,
                            header=0,
                            low_memory=False
                        )
            paws_df.columns = paws_df.columns.str.strip()
            paws_df.rename(columns={'mon':'month', 'min':'minute'}, inplace=True)
            paws_df['date'] = pd.to_datetime(paws_df[['year', 'month', 'day', 'hour', 'minute']])
            
            paws_df.drop( # no reference data for these col's OR no data exists in these columns (e.g. sth) OR scalar (e.g. alt)
                        ['int', 'sth_temp', 'sth_hum', 'bmp2_alt', 'vis_light', 'ir_light', 'uv_light',
                         'year', 'month', 'day', 'hour', 'minute'], 
                        axis=1, 
                        inplace=True
                    )
            
            paws_df = paws_df[['date', 'bmp2_temp', 'htu_temp', 'mcp9808', 'bme2_hum', 'htu_hum', 'bmp2_pres',
                               'bmp2_slp', 'wind_dir', 'wind_speed', "tipping"]]
            
            paws_df['year_month'] = paws_df['date'].dt.to_period('M')
            paws_df['year_month_day'] = paws_df['date'].dt.to_period('D')
            paws_df['year_month_day_hour'] = paws_df['date'].dt.to_period('H')
            
            paws_df.set_index('date', inplace=True)

            paws_dfs.append(paws_df)

        else:
            tsms_df = pd.read_csv(
                            data_origin+station_directories[i]+file,
                            header=0,
                            low_memory=False
                        )
            tsms_df.columns = tsms_df.columns.str.strip()
            tsms_df.rename(columns={'mon':'month', 'min':'minute'}, inplace=True)
            tsms_df['date'] = pd.to_datetime(tsms_df[['year', 'month', 'day', 'hour', 'minute']])
            
            tsms_df.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1, inplace=True)

            tsms_df = tsms_df[['date', 'temp', 'humidity', 'actual_pressure', 'sea_level_pressure', 
                               'avg_wind_dir', 'avg_wind_speed', 'total_rainfall']]
            
            tsms_df['year_month'] = tsms_df['date'].dt.to_period('M')
            tsms_df['year_month_day'] = tsms_df['date'].dt.to_period('D')
            tsms_df['year_month_day_hour'] = tsms_df['date'].dt.to_period('H')

            tsms_df.set_index('date', inplace=True) 

            tsms_dfs.append(tsms_df) 

    
    # Ignore warnings so they don't clog up the console output
    warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.")

    """
    =============================================================================================================================
                Eliminate outliers from the dataset and store in separate dataframe for each station.
            Creates a FILTERED dataframe, where all cells containing an outlier have been replaced with np.nan
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

    """
    =============================================================================================================================
    Phase 0: Fill in any data gaps w/ empty rows corresponding to the missing timestamp
    =============================================================================================================================
    """
    print(f"Filling in data gaps for PAWS station {station_directories[i][8:14]}")
    paws_df_FILTERED = functions.fill_empty_rows(paws_df_FILTERED, timedelta(minutes=1))
    print(f"Filling in data gaps for TSMS station {station_directories[i][8:14]}")
    tsms_df_FILTERED = functions.fill_empty_rows(tsms_df_FILTERED, timedelta(minutes=1))

    paws_df_FILTERED.to_csv(f"/Users/rzieber/Documents/3D-PAWS/Turkiye/output/{station_directories[i]}/{station_directories[i][8:14]}_Complete_Record_[DATA-GAPS-FILLED].csv")
    tsms_df_FILTERED.to_csv(f"/Users/rzieber/Documents/3D-PAWS/Turkiye/output/{station_directories[i]}/{station_directories[i][8:14]}_Complete_Record_[DATA-GAPS-FILLED]-TSMS.csv")

#     print()
#     print("Starting outlier removal for ", station_directories[i][8:14])
#     """
#     =============================================================================================================================
#     Phase 1: Filter out nulls (-999.99)
#     =============================================================================================================================
#     """
#     print("Phase 1: Filtering out nulls (-999.99).")

#     for variable in variable_mapper:                        # Filtering TSMS ----------------------------------------------------
#         mask_999 = tsms_df_FILTERED[variable] == -999.99    # Replace -999.99 with np.nan
#         tsms_df_FILTERED.loc[mask_999, variable] = np.nan
        
#         mask_null = tsms_df_FILTERED[variable].isnull()     # Identify nulls
        
#         if mask_null.any():                                 # Add nulls to outliers_df
#             outliers_to_add = pd.DataFrame({
#                 'date': tsms_df_FILTERED['date'][mask_null],
#                 'column_name': variable,
#                 'original_value': -999.99,
#                 'outlier_type': 'null'
#             })
#             tsms_outliers = pd.concat([tsms_outliers, outliers_to_add], ignore_index=True)
        

#         outliers_to_add = None
        

#         for var in variable_mapper[variable]:               # Filtering PAWS ----------------------------------------------------
#             mask_999 = paws_df_FILTERED[var] == -999.99   
#             paws_df_FILTERED.loc[mask_999, var] = np.nan

#             mask_null = paws_df_FILTERED[var].isnull()    

#             if mask_null.any():                  
#                 outliers_to_add = pd.DataFrame({
#                     'date': paws_df_FILTERED['date'][mask_null],
#                     'column_name': var,
#                     'original_value': -999.99,
#                     'outlier_type': 'null'
#                 })
#                 paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)
    
#     """
#     =============================================================================================================================
#     Phase 2: Filter out time resets
#     =============================================================================================================================
#     """
#     print(f"Phase 2: Filtering out time resets.")

#     tsms_df_FILTERED = tsms_df_FILTERED.sort_values(by='date')  # Filtering TSMS ------------------------------------------------
#     tsms_df_FILTERED['time_diff'] = tsms_df_FILTERED['date'].diff()
    
#     out_of_order_tsms = tsms_df_FILTERED[tsms_df_FILTERED['time_diff'] <= pd.Timedelta(0)]  # Identify out-of-order timestamps

#     tsms_outliers = pd.concat([tsms_outliers, pd.DataFrame({    # Add out-of-order timestamps to outlier dataframe
#         'date': out_of_order_tsms['date'],
#         'column_name': 'date',
#         'original_value': np.nan,
#         'outlier_type': 'timestamp reset'
#     })], ignore_index=True)

#     df_sequential = tsms_df_FILTERED[tsms_df_FILTERED['time_diff'] > pd.Timedelta(0)]       # Filter out-of-order timestamps
#     tsms_df_FILTERED = df_sequential.drop(columns=['time_diff'])


#     df_sequential = None  


#     paws_df_FILTERED = paws_df_FILTERED.sort_values(by='date')  # Filtering PAWS ------------------------------------------------
#     paws_df_FILTERED['time_diff'] = paws_df_FILTERED['date'].diff()
    
#     out_of_order_paws = paws_df_FILTERED[paws_df_FILTERED['time_diff'] <= pd.Timedelta(0)]  # Identify out-of-order timestamps

#     paws_outliers = pd.concat([paws_outliers, pd.DataFrame({    # Add out-of-order timestamps to outlier dataframe
#         'date': out_of_order_paws['date'],
#         'column_name': 'date',
#         'original_value': np.nan,
#         'outlier_type': 'timestamp reset'
#     })], ignore_index=True)

#     df_sequential = paws_df_FILTERED[paws_df_FILTERED['time_diff'] > pd.Timedelta(0)]       # Filter out-of-order timestamps
#     paws_df_FILTERED = df_sequential.drop(columns=['time_diff'])


#     """
#     =============================================================================================================================
#     Phase 3: Filter out unrealistic values
#     =============================================================================================================================
#     """
#     print(f"Phase 3: Filtering out unrealistic values.") 

#     for variable in variable_mapper:                                # Filtering TSMS --------------------------------------------
#         existing_nulls = tsms_df_FILTERED[variable].isnull()        # Create a mask for existing nulls before applying the filter
        
#         if variable == "temp":
#             mask_temp = (tsms_df_FILTERED[variable] > 50) | (tsms_df_FILTERED[variable] < -50)
#             tsms_df_FILTERED.loc[mask_temp, variable] = np.nan
#         elif variable == "humidity":
#             mask_hum = (tsms_df_FILTERED[variable] > 100) | (tsms_df_FILTERED[variable] < 0)
#             tsms_df_FILTERED.loc[mask_hum, variable] = np.nan
#         elif variable == "actual_pressure" or variable == "sea_level_pressure":
#             mask_pres = (tsms_df_FILTERED[variable] > 1084) | (tsms_df_FILTERED[variable] < 870)
#             tsms_df_FILTERED.loc[mask_pres, variable] = np.nan
#         elif variable == "avg_wind_speed":
#             mask_wndspd = (tsms_df_FILTERED[variable] > 100) | (tsms_df_FILTERED[variable] < 0)
#             tsms_df_FILTERED.loc[mask_wndspd, variable] = np.nan
#         elif variable == "avg_wind_dir":
#             mask_wnddir = (tsms_df_FILTERED[variable] > 360) | (tsms_df_FILTERED[variable] < 0)
#             tsms_df_FILTERED.loc[mask_wnddir, variable] = np.nan
#         else: # total_rainfall
#             mask_rain = (tsms_df_FILTERED[variable] > 32) | (tsms_df_FILTERED[variable] < 0)
#             tsms_df_FILTERED.loc[mask_rain, variable] = np.nan
        
#         new_nulls = tsms_df_FILTERED[variable].isnull() & ~existing_nulls   # Create a mask for new nulls after applying the filter
        
#         outliers_to_add = pd.DataFrame({                                # Add new nulls to the outlier dataframe
#             'date': tsms_df_FILTERED.loc[new_nulls, 'date'],
#             'column_name': variable,
#             'original_value': tsms_df_FILTERED.loc[new_nulls, variable],
#             'outlier_type': 'threshold'
#         })
#         tsms_outliers = pd.concat([tsms_outliers, outliers_to_add], ignore_index=True)
        

#         outliers_to_add = None
#         new_nulls = None
        

#         for var in variable_mapper[variable]:               # Filtering PAWS ----------------------------------------------------
#             existing_nulls = paws_df_FILTERED[var].isnull()

#             if var == "bmp2_temp" or var == "htu_temp" or var == "mcp9808":
#                 mask_temp = (paws_df_FILTERED[var] > 50) | (paws_df_FILTERED[var] < -50)
#                 paws_df_FILTERED.loc[mask_temp, var] = np.nan
#             elif var == "htu_hum" or var == "bme2_hum":
#                 mask_hum = (paws_df_FILTERED[var] > 100) | (paws_df_FILTERED[var] < 0)
#                 paws_df_FILTERED.loc[mask_hum, var] = np.nan
#             elif var == "bmp2_pres" or var == "bmp2_slp":
#                 mask_pres = (paws_df_FILTERED[var] > 1084) | (paws_df_FILTERED[var] < 870)
#                 paws_df_FILTERED.loc[mask_pres, var] = np.nan
#             elif var == "wind_speed":
#                 paws_df_FILTERED[var] = pd.to_numeric(paws_df_FILTERED[var], errors='coerce')
#                 mask_wndspd = (paws_df_FILTERED[var] > 100) | (paws_df_FILTERED[var] < 0)
#                 paws_df_FILTERED.loc[mask_wndspd, var] = np.nan
#             elif var == "wind_dir":
#                 mask_wnddir = (paws_df_FILTERED[var] > 360) | (paws_df_FILTERED[var] < 0)
#                 paws_df_FILTERED.loc[mask_wnddir, var] = np.nan
#             elif var == "tipping":
#                 mask_rain = (paws_df_FILTERED[var] > 32) | (paws_df_FILTERED[var] < 0)
#                 paws_df_FILTERED.loc[mask_rain, var] = np.nan
#             else:
#                 print("Error with ", var)
            
#             new_nulls = paws_df_FILTERED[var].isnull() & ~existing_nulls  
            
#             outliers_to_add = pd.DataFrame({               
#                 'date': paws_df_FILTERED.loc[new_nulls, 'date'],
#                 'column_name': var,
#                 'original_value': paws_df_FILTERED.loc[new_nulls, var],
#                 'outlier_type': 'threshold'
#             })
#             paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)

#     """
#     =============================================================================================================================
#     =============================================================================================================================
#     =============================================================================================================================
#     Phase 4: Filter out HTU bit-swtich from 3D-PAWS dataset
#     =============================================================================================================================
#     =============================================================================================================================
#     =============================================================================================================================
#     """
#     print(f"Phase 4: Filtering out contextual outliers.")

#     # Ensure 'date' column is converted to datetime and set as index
#     if not isinstance(paws_df_FILTERED.index, pd.DatetimeIndex):
#         paws_df_FILTERED['date'] = pd.to_datetime(paws_df_FILTERED['date'], errors='coerce')
#         paws_df_FILTERED.set_index('date', inplace=True)

#     start = datetime(2022, 12, 1, 0, 0, 0)
#     end = datetime(2022, 12, 2, 23, 59, 59)

#     for variable in variable_mapper:
#         if variable != "humidity": continue

#         for var in variable_mapper[variable]: # check 3D-PAWS's HTU
#             threshold = 1.2
#             last_non_null = None
#             diff_outliers = []

#             paws_df_FILTERED = paws_df_FILTERED.loc[start:end]
#             paws_df_FILTERED.to_csv("/Users/rzieber/Downloads/imfuckingtired.csv")

#             # BEFORE ---------------
#             for year_month, paws_group in paws_df_FILTERED.groupby('year_month'):
#                 tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

#                 plt.figure(figsize=(20, 12))

#                 plt.plot(paws_group.index, paws_group['htu_hum'], marker='.', markersize=1, label=f"3D PAWS htu_hum")
#                 plt.plot(tsms_group.index, tsms_group['humidity'], marker='.', markersize=1, label='TSMS')
            
#                 plt.title(f'{station_directories[i][8:14]} Humidity for {year_month}: 3D PAWS htu_hum versus TSMS')
#                 plt.xlabel('Date')
#                 plt.ylabel('Humidity (%)')
#                 plt.xticks(rotation=45)

#                 plt.legend()

#                 plt.grid(True)
#                 plt.tight_layout()
#                 plt.savefig("/Users/rzieber/Documents/GitHub/Turkiye_Intercomparison/plots/station_TSMS00/humidity/raw/before.png")        

#                 plt.clf()
#                 plt.close()
#             # ------------------------

#             paws_df_FILTERED.reset_index(drop=True, inplace=True)   # Reset the index to ensure it is a simple range

#             i = 0
#             while i < len(paws_df_FILTERED) - 2:
#                 if pd.isna(last_non_null): print("Error with ", last_non_null, ". Current index: ", i)
                
#                 if pd.notna(paws_df_FILTERED.loc[i, var]): 
#                     last_non_null = paws_df_FILTERED.loc[i, var]
#                 if pd.isna(paws_df_FILTERED.loc[i+1, var]): 
#                     i += 1
#                     continue

#                 if pd.isna(last_non_null) and i == 0: continue

#                 diff = abs(paws_df_FILTERED.loc[i+1, var] - last_non_null)
                
#                 if diff is None or last_non_null is None: 
#                     print("Diff: ", diff, "\t\tType: ", type(diff))
#                     print("Last non null: ", last_non_null, "\t\tType: ", type(last_non_null))
#                     print(i)
#                     i += 1
#                     continue

#                 if diff > threshold: 
#                     diff_outliers.append({
#                         'date': paws_df_FILTERED.loc[i+1, 'date'],
#                         'column_name': var,
#                         'original_value': paws_df_FILTERED.loc[i+1, var],
#                         'outlier_type': 'diff contextual'
#                     })

#                     paws_df_FILTERED.loc[i+1, var] = np.nan

#                 i += 1
            
#             diff_outliers_df = pd.DataFrame(diff_outliers)

#             paws_outliers = pd.concat([paws_outliers, diff_outliers_df], ignore_index=True)

#             paws_outliers.to_csv("/Users/rzieber/Downloads/TSMS00_htu_test.csv")

#             paws_df_FILTERED.set_index("date", inplace=True)

#             # AFTER ---------------
#             for year_month, paws_group in paws_df_FILTERED.groupby('year_month'):
#                 tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

#                 plt.figure(figsize=(20, 12))

#                 plt.plot(paws_group.index, paws_group['htu_hum'], marker='.', markersize=1, label=f"3D PAWS htu_hum")
#                 plt.plot(tsms_group.index, tsms_group['humidity'], marker='.', markersize=1, label='TSMS')
            
#                 plt.title(f'{station_directories[i][8:14]} Humidity for {year_month}: 3D PAWS htu_hum versus TSMS')
#                 plt.xlabel('Date')
#                 plt.ylabel('Humidity (%)')
#                 plt.xticks(rotation=45)

#                 plt.legend()

#                 plt.grid(True)
#                 plt.tight_layout()
#                 plt.savefig("/Users/rzieber/Documents/GitHub/Turkiye_Intercomparison/plots/station_TSMS00/humidity/raw/after.png")  
            
#                 plt.clf()
#                 plt.close()
#             # ---------------------


# # ----------------------------------------------------------------------------------------------------------------------------------------------- PLOT GEN LOGIC
    
#     # """
#     # =============================================================================================================================
#     # Create a time series plot of the 3D PAWS station data versus the TSMS reference station. MONTHLY RECORDS
#     # =============================================================================================================================
#     # """
#     # print(f"{station_directories[i][8:14]}: " \
#     #                    "Time-series plots for temperature, humidity, actual pressure and sea level pressure -- monthly records")
    
#     # paws_df_FILTERED.set_index('date', inplace=True)
#     # tsms_df_FILTERED.set_index('date', inplace=True)
    
#     # tsms_df_FILTERED = tsms_df_FILTERED[paws_df_FILTERED.index[0]:paws_df_FILTERED.index[-1]]

#     # # Humidity ------------------------------------------------------------------------------------------------------------------
#     # print("Humidity -- raw")
#     # for h in variable_mapper['humidity']:
#     #     for year_month, paws_group in paws_df_FILTERED.groupby('year_month'):
#     #         tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

#     #         plt.figure(figsize=(20, 12))

#     #         plt.plot(paws_group.index, paws_group[f'{h}'], marker='.', markersize=1, label=f"3D PAWS {h}")
#     #         plt.plot(tsms_group.index, tsms_group['humidity'], marker='.', markersize=1, label='TSMS')
        
#     #         plt.title(f'{station_directories[i][8:14]} Humidity for {year_month}: 3D PAWS {h} versus TSMS')
#     #         plt.xlabel('Date')
#     #         plt.ylabel('Humidity (%)')
#     #         plt.xticks(rotation=45)

#     #         plt.legend()

#     #         plt.grid(True)
#     #         plt.tight_layout()
#     #         plt.savefig("tsms00_htu_filtering.png")            
#     #         plt.clf()
#     #         plt.close()
