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
from ...resources import functions
from datetime import timedelta

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

data_origin = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/output/"
#data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/time_series/complete_records/"
#data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/time_series/monthly_plots/"
#data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/time_series_IQR-Filter/monthly_plots/"
#data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/time_series_Z_and_IQR/monthly_plots/"
data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/time_series_Z_and_IQR_and_Diff/monthly_plots/"
#data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/box_plots/monthly_plots/"
#data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/box_plots_IQR-Filter/monthly_plots/"
#data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/box_plots_Z_and_IQR/monthly_plots/"
#data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/box_plots_Z_and_IQR_and_Diff/monthly_plots/"
#data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/time_series/temp_comparison/3DPAWS_and_TSMS/monthly_plots_2/"
#data_destination = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/plots/time_series/temp_comparison/3DPAWS_Only/monthly_plots/"

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

    """
    =============================================================================================================================
    Phase 0: Fill in any data gaps w/ empty rows corresponding to the missing timestamp
    =============================================================================================================================
    """
    print(f"Filling in data gaps for PAWS station {station_directories[i][8:14]}")
    paws_df_FILTERED = functions.fill_empty_rows(paws_df_FILTERED, timedelta(minutes=1))
    print(f"Filling in data gaps for TSMS station {station_directories[i][8:14]}")
    tsms_df_FILTERED = functions.fill_empty_rows(tsms_df_FILTERED, timedelta(minutes=1))

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
                'outlier_type': 'null'
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
                    'outlier_type': 'null'
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
        'outlier_type': 'timestamp reset'
    })], ignore_index=True)

    df_sequential = tsms_df_FILTERED[tsms_df_FILTERED['time_diff'] > pd.Timedelta(0)]       # Filter out-of-order timestamps
    tsms_df_FILTERED = df_sequential.drop(columns=['time_diff'])


    df_sequential = None  


    paws_df_FILTERED = paws_df_FILTERED.sort_values(by='date')  # Filtering PAWS ------------------------------------------------
    paws_df_FILTERED['time_diff'] = paws_df_FILTERED['date'].diff()
    
    out_of_order_paws = paws_df_FILTERED[paws_df_FILTERED['time_diff'] <= pd.Timedelta(0)]  # Identify out-of-order timestamps

    paws_outliers = pd.concat([paws_outliers, pd.DataFrame({    # Add out-of-order timestamps to outlier dataframe
        'date': out_of_order_paws['date'],
        'column_name': 'date',
        'original_value': np.nan,
        'outlier_type': 'timestamp reset'
    })], ignore_index=True)

    df_sequential = paws_df_FILTERED[paws_df_FILTERED['time_diff'] > pd.Timedelta(0)]       # Filter out-of-order timestamps
    paws_df_FILTERED = df_sequential.drop(columns=['time_diff'])


    """
    =============================================================================================================================
    Phase 3: Filter out unrealistic values
    =============================================================================================================================
    """
    print(f"Phase 3: Filtering out unrealistic values.") 

    for variable in variable_mapper:                                # Filtering TSMS --------------------------------------------
        existing_nulls = tsms_df_FILTERED[variable].isnull()        # Create a mask for existing nulls before applying the filter
        
        if variable == "temp":
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
        else: # total_rainfall
            mask_rain = (tsms_df_FILTERED[variable] > 32) | (tsms_df_FILTERED[variable] < 0)
            tsms_df_FILTERED.loc[mask_rain, variable] = np.nan
        
        new_nulls = tsms_df_FILTERED[variable].isnull() & ~existing_nulls   # Create a mask for new nulls after applying the filter
        
        outliers_to_add = pd.DataFrame({                                # Add new nulls to the outlier dataframe
            'date': tsms_df_FILTERED.loc[new_nulls, 'date'],
            'column_name': variable,
            'original_value': tsms_df_FILTERED.loc[new_nulls, variable],
            'outlier_type': 'threshold'
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
                'outlier_type': 'threshold'
            })
            paws_outliers = pd.concat([paws_outliers, outliers_to_add], ignore_index=True)

    paws_df_FILTERED.to_csv(f"/Users/rzieber/Downloads/fill_gaps_paws_{station_directories[i][8:14]}.csv")
    tsms_df_FILTERED.to_csv(f"/Users/rzieber/Downloads/fill_gaps_tsms_{station_directories[i][8:14]}.csv")

    # """
    # =============================================================================================================================
    # Phase 4: Filter out contextual outliers
    # =============================================================================================================================
    # """
    # print(f"Phase 4: Filtering out contextual outliers.")

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


    #     # Abs Diff --------------
    #     print("\t\tAbsolute Diff")

    #     threshold = None
    #     last_non_null = None
    #     diff_outliers = []

    #     i = 0
    #     while i < len(tsms_df_FILTERED) - 2:
    #         if pd.isna(last_non_null): print("Error with ", last_non_null, ". Current index: ", i)
    #         if pd.notna(tsms_df_FILTERED.loc[i, variable]): 
    #             last_non_null = tsms_df_FILTERED.loc[i, variable]
    #         if pd.isna(tsms_df_FILTERED.loc[i+1, variable]): 
    #             i += 1
    #             continue

    #         if pd.isna(last_non_null) and i == 0: continue

    #         if variable == "temp": threshold = 2
    #         elif variable == "humidity": threshold = 2
    #         elif variable in ["actual_pressure", "sea_level_pressure"]: threshold = 2

    #         diff = abs(tsms_df_FILTERED.loc[i+1, variable] - last_non_null)

    #         if diff is None or last_non_null is None: 
    #                 print("Diff: ", diff, "\t\tType: ", type(diff))
    #                 print("Last non null: ", last_non_null, "\t\tType: ", type(last_non_null))
    #                 print("Index: ", i)
    #                 continue

    #         if diff > threshold: 
    #             diff_outliers.append({
    #                 'date': tsms_df_FILTERED.loc[i+1, 'date'],
    #                 'column_name': variable,
    #                 'original_value': tsms_df_FILTERED.loc[i+1, variable],
    #                 'outlier_type': 'diff contextual'
    #             })

    #             tsms_df_FILTERED.loc[i+1, variable] = np.nan

    #         i += 1
        
    #     diff_outliers_df = pd.DataFrame(diff_outliers)
    #     # -----------------------         
        
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


    # print()
    # # paws_df_FILTERED.to_csv(f"/Users/rzieber/Downloads/paws_FILTERED_{station_directories[i][8:14]}.csv")
    # # tsms_df_FILTERED.to_csv(f"/Users/rzieber/Downloads/tsms_FILTERED_{station_directories[i][8:14]}.csv")
    # paws_outliers.to_csv(f"/Users/rzieber/Downloads/paws_outliers_{station_directories[i][8:14]}.csv")
    # tsms_outliers.to_csv(f"/Users/rzieber/Downloads/tsms_outliers_{station_directories[i][8:14]}.csv")

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
    # for t in variable_mapper['temp']:
    #     for year_month, paws_group in paws_df_FILTERED.groupby('year_month'):
    #         tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]

    #         plt.figure(figsize=(20, 12))

    #         plt.plot(paws_group.index, paws_group[f'{t}'], marker='.', markersize=1, label=f"3D PAWS {t}")
    #         plt.plot(tsms_group.index, tsms_group['temp'], marker='.', markersize=1, label='TSMS')
        
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
    Creating box plots for temperature, relative humidity, and pressure. COMPLETE RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]} Box Plots")

    # paws_temp_cols = ['bmp2_temp', 'htu_temp', 'mcp9808']
    # paws_hum_cols = ['htu_hum', 'bme2_hum']
    # paws_pres_cols = ['bmp2_pres']
    # tsms_temp_cols = ['temp']
    # tsms_hum_cols = ['humidity']
    # tsms_pres_cols = ['actual_pressure']

    # combined_df = paws_df_FILTERED.merge(tsms_df_FILTERED[tsms_temp_cols + tsms_hum_cols + tsms_pres_cols], left_index=True, right_index=True)
    # combined_temp_df = combined_df[paws_temp_cols + ['temp']].copy()
    # combined_hum_df = combined_df[paws_hum_cols + ['humidity']].copy()
    # combined_pres_df = combined_df[paws_pres_cols + ['actual_pressure']].copy()

    # # temperature box plot
    # plt.figure(figsize=(20, 12))
    # sns.boxplot(data=combined_temp_df)

    # plt.title(f'{station_directories[i][8:14]} Temperature Box Plot: 3D-PAWS & TSMS')
    # plt.xlabel('Temperature Sensors')
    # plt.ylabel('Temperature (°C)')
    # plt.xticks([0, 1, 2, 3], ['BMP2 Temp', 'HTU Temp', 'MCP9808 Temp', 'TSMS Reference Temp'])
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"/temp/raw/IQR_stats_{station_directories[i][8:14]}.png")
    # plt.clf()
    # plt.close()

    # # humidity boxplot
    # plt.figure(figsize=(20, 12))
    # sns.boxplot(data=combined_hum_df)

    # plt.title(f'{station_directories[i][8:14]} Humidity Box Plot: 3D-PAWS & TSMS')
    # plt.xlabel('Humidity Sensors')
    # plt.ylabel('Humidity (%)')
    # plt.xticks([0, 1, 2], ['HTU Hum', 'BME2 Hum', 'TSMS Reference Hum'])
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"/humidity/raw/IQR_stats_{station_directories[i][8:14]}.png")
    # plt.clf()
    # plt.close()

    # # pressure box plots
    # plt.figure(figsize=(20, 12))
    # sns.boxplot(data=combined_pres_df)

    # plt.title(f'{station_directories[i][8:14]} Pressure Box Plot: 3D-PAWS & TSMS')
    # plt.xlabel('Pressure Sensors')
    # plt.ylabel('Station Pressure (hPa)')
    # plt.xticks([0, 1], ['BMP2 Pressure', 'TSMS Reference Pressure'])
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"/actual_pressure/raw/IQR_stats_{station_directories[i][8:14]}.png")
    # plt.clf()
    # plt.close()



    
    """
    =============================================================================================================================
    Create a time series plot of the 3D PAWS station data versus the TSMS reference station. COMPLETE RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][8:14]}: " \
    #                     "Time-series plots for temperature, humidity, actual & sea level pressure, and cumulative rainfall -- complete records")
    
    # tsms_FILTERED = tsms_df[paws_df.index[0]:paws_df.index[-1]]

    # Temperature ---------------------------------------------------------------------------------------------------------------
    # print("Temperature -- raw")
    # for t in variable_mapper['temp']:
    #     plt.figure(figsize=(20, 12))

    #     plt.plot(paws_df.index, paws_df[f'{t}'], marker='.', markersize=1, label=f"3D PAWS {t}")
    #     plt.plot(tsms_FILTERED.index, tsms_FILTERED['temp'], marker='.', markersize=1, alpha=0.25, label='TSMS')
    
    #     plt.title(f'{station_directories[i][8:14]} Temperature: 3D PAWS {t} versus TSMS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Temperature (˚C)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(data_destination+station_directories[i]+f"temp/raw/{station_directories[i][8:14]}_{t}.png")

    #     plt.clf()
    #     plt.close()

    # print("Temperature -- nulls filtered")
    # for t in variable_mapper['temp']:
    #     paws_df_FILTERED = paws_df[paws_df[f"{t}"] > -999.99]

    #     plt.figure(figsize=(20, 12))

    #     plt.plot(paws_df_FILTERED.index, paws_df_FILTERED[f'{t}'], marker='.', markersize=1, label=f"3D PAWS {t}")
    #     plt.plot(tsms_FILTERED.index, tsms_FILTERED['temp'], alpha=0.5, marker='.', markersize=1, label='TSMS')
    
    #     plt.title(f'{station_directories[i][8:14]} Temperature: 3D PAWS {t} versus TSMS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Temperature (˚C)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(data_destination+station_directories[i]+f"temp/filtered_nulls/{station_directories[i][8:14]}_{t}_FILTERED.png")
        
    #     plt.clf()
    #     plt.close()

    # # Humidity ------------------------------------------------------------------------------------------------------------------
    # print("Humidity -- raw")
    # for h in variable_mapper['humidity']:
    #     plt.figure(figsize=(20, 12))

    #     plt.plot(paws_df.index, paws_df[f'{h}'], marker='.', markersize=1, label=f"3D PAWS {h}")
    #     plt.plot(tsms_FILTERED.index, tsms_FILTERED['humidity'], marker='.', markersize=1, label='TSMS')
    
    #     plt.title(f'{station_directories[i][8:14]} Humidity: 3D PAWS {h} versus TSMS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Humidity (%)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(data_destination+station_directories[i]+f"humidity/raw/{station_directories[i][8:14]}_{h}.png")
        
    #     plt.clf()
    #     plt.close()

    # print("Humidity -- nulls filtered")
    # for h in variable_mapper['humidity']:
    #     paws_df_FILTERED = paws_df[paws_df[f"{h}"] > -999.99]

    #     plt.figure(figsize=(20, 12))

    #     plt.plot(paws_df_FILTERED.index, paws_df_FILTERED[f'{h}'], marker='.', markersize=1, label=f"3D PAWS {h}")
    #     plt.plot(tsms_FILTERED.index, tsms_FILTERED['temp'], marker='.', markersize=1, label='TSMS')
    
    #     plt.title(f'{station_directories[i][8:14]} Humidity: 3D PAWS {h} versus TSMS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Humidity (%)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(data_destination+station_directories[i]+f"humidity/filtered_nulls/{station_directories[i][8:14]}_{h}_FILTERED.png")
        
    #     plt.clf()
    #     plt.close()

    
    # # Pressure ------------------------------------------------------------------------------------------------------------------
    # print("Actual pressure -- raw")
    # plt.figure(figsize=(20, 12))

    # plt.plot(paws_df.index, paws_df['bmp2_pres'], marker='.', markersize=1, label="3D PAWS bmp2_pres")
    # plt.plot(tsms_FILTERED.index, tsms_FILTERED['actual_pressure'], marker='.', markersize=1, label='TSMS')

    # plt.title(f'{station_directories[i][8:14]} Pressure: 3D PAWS bmp2_pres versus TSMS')
    # plt.xlabel('Date')
    # plt.ylabel('Pressure (mbar)')
    # plt.xticks(rotation=45)

    # plt.legend()

    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"actual_pressure/raw/{station_directories[i][8:14]}_bmp2_pressure.png")
    
    # plt.clf()
    # plt.close()


    # print("Actual pressure -- nulls filtered")
    # paws_df_FILTERED = paws_df[paws_df["bmp2_pres"] > -999.99] 
    # tsms_df_FILTERED = tsms_FILTERED[(tsms_FILTERED['actual_pressure']) > 0]

    # plt.figure(figsize=(20, 12))

    # plt.plot(paws_df_FILTERED.index, paws_df_FILTERED['bmp2_pres'], marker='.', markersize=1, label="3D PAWS bmp2_pres")
    # plt.plot(tsms_df_FILTERED.index, tsms_df_FILTERED['actual_pressure'], marker='.', markersize=1, label='TSMS')

    # plt.title(f'{station_directories[i][8:14]} Pressure: 3D PAWS bmp2_pres versus TSMS')
    # plt.xlabel('Date')
    # plt.ylabel('Pressure (mbar)')
    # plt.xticks(rotation=45)

    # plt.legend()

    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"actual_pressure/filtered_nulls/{station_directories[i][8:14]}_bmp2_pressure_FILTERED.png")
    
    # plt.clf()
    # plt.close()


    # # Sea Level Pressure --------------------------------------------------------------------------------------------------------
    # print("Sea level pressure -- raw")
    # plt.figure(figsize=(20, 12)) 

    # plt.plot(paws_df.index, paws_df['bmp2_slp'], marker='.', markersize=1, label="3D PAWS bmp2_slp")
    # plt.plot(tsms_FILTERED.index, tsms_FILTERED['sea_level_pressure'], marker='.', markersize=1, label='TSMS')

    # plt.title(f'{station_directories[i][8:14]} Sea Level Pressure: 3D PAWS bmp2_slp versus TSMS')
    # plt.xlabel('Date')
    # plt.ylabel('Pressure (mbar)')
    # plt.xticks(rotation=45)

    # plt.legend()

    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"sea_level_pressure/raw/{station_directories[i][8:14]}_bmp2_slp.png")
    
    # plt.clf()
    # plt.close()


    # print("Sea level pressure -- nulls filtered")
    # paws_df_FILTERED = paws_df[paws_df["bmp2_slp"] > -999.99] 
    # tsms_df_FILTERED = tsms_FILTERED[(tsms_FILTERED['sea_level_pressure']) > 0]

    # plt.figure(figsize=(20, 12))

    # plt.plot(paws_df_FILTERED.index, paws_df_FILTERED['bmp2_slp'], marker='.', markersize=1, label="3D PAWS bmp2_slp")
    # plt.plot(tsms_df_FILTERED.index, tsms_df_FILTERED['sea_level_pressure'], marker='.', markersize=1, label='TSMS')

    # plt.title(f'{station_directories[i][8:14]} Sea Level Pressure: 3D PAWS bmp2_slp versus TSMS')
    # plt.xlabel('Date')
    # plt.ylabel('Pressure (mbar)')
    # plt.xticks(rotation=45)

    # plt.legend()

    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"sea_level_pressure/filtered_nulls/{station_directories[i][8:14]}_bmp2_slp_FILTERED.png")
    
    # plt.clf()
    # plt.close()


    # # Rainfall Accumulation --------------------------------------------------------------------------------------------------------
    # print("Rainfall accumulation -- nulls filtered")
    # for paws_station in station_directories: 
    #     paws_df_FILTERED = paws_df[paws_df['tipping'] >= 0].copy().reset_index()
    #     tsms_df_FILTERED = tsms_df[(tsms_df['total_rainfall']) >= 0].copy().reset_index()

    #     if station_directories[i] == "station_TSMS08/":
    #         exclude_garbage_start = pd.to_datetime("2023-04-08")
    #         exclude_garbage_end = pd.to_datetime("2023-04-15")
    #         paws_df_FILTERED = paws_df_FILTERED[
    #             ~((paws_df_FILTERED['date'] >= exclude_garbage_start) & (paws_df_FILTERED['date'] <= exclude_garbage_end))
    #         ]
    #     elif station_directories[i] == "station_TSMS04/":
    #         exclude_garbage = pd.to_datetime("2022-09-01")
    #         tsms_df_FILTERED = tsms_df_FILTERED[
    #             (tsms_df_FILTERED['date'] >= exclude_garbage) & \
    #             (tsms_df_FILTERED['date'] <= paws_df_FILTERED['date'].iloc[-1])
    #         ]
    #         paws_df_FILTERED = paws_df_FILTERED[
    #             (paws_df_FILTERED['date'] >= exclude_garbage) & \
    #             (paws_df_FILTERED['date'] <= paws_df_FILTERED['date'].iloc[-1])
    #         ]

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
    #     plt.savefig(data_destination+station_directories[i]+f"total_rainfall/filtered_nulls/{station_directories[i][8:14]}_rainfall_accumulation_FILTERED.png")
        
    #     plt.clf()
    #     plt.close()


    



    """
    =============================================================================================================================
    Create time series plots of the 3 temperature sensors compared to TSMS. COMPLETE RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][:len(station_directories[i])-1]} Temperature Comparison")

    # tsms_FILTERED = tsms_df[paws_df.index[0]:paws_df.index[-1]]

    # paws_df_FILTERED = paws_df[
    #     (paws_df["bmp2_temp"] > -50) | (paws_df["htu_temp"] > -50) | (paws_df["mcp9808"] > -50)
    #     ]

    # plt.figure(figsize=(20, 12))

    # plt.plot(paws_df_FILTERED.index, paws_df_FILTERED["bmp2_temp"], marker='.', markersize=1, label=f"3D PAWS bmp2_temp")
    # plt.plot(paws_df_FILTERED.index, paws_df_FILTERED["htu_temp"], marker='.', markersize=1, label=f"3D PAWS htu_temp")
    # plt.plot(paws_df_FILTERED.index, paws_df_FILTERED["mcp9808"], marker='.', markersize=1, label=f"3D PAWS mcp9808")
    # plt.plot(tsms_FILTERED.index, tsms_FILTERED['temp'], alpha=0.5, marker='.', markersize=1, label='TSMS')

    # plt.title(f'{station_directories[i][8:14]} Temperature Comparison: 3D PAWS versus TSMS')
    # #plt.title(f'{station_directories[i][8:14]} Temperature Comparison: 3D PAWS')
    # plt.xlabel('Date')
    # plt.ylabel('Temperature (˚C)')
    # plt.xticks(rotation=45)

    # plt.legend()

    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(data_destination+station_directories[i]+f"temp/filtered_nulls/{station_directories[i][8:14]}_temp_comparison_FILTERED.png")
    
    # plt.clf()
    # plt.close()


    """
    =============================================================================================================================
    Create time series plots of the 3 temperature sensors compared to TSMS. MONTHLY RECORDS
    =============================================================================================================================
    """
    # print(f"{station_directories[i][:len(station_directories[i])-1]} Temperature Comparison -- nulls filtered")

    # tsms_df_FILTERED = tsms_df[paws_df.index[0]:paws_df.index[-1]]

    # paws_df_FILTERED = paws_df.replace(-999.99, np.nan)
    #tsms_df_FILTERED = tsms_df_FILTERED.replace(-999.99, np.nan, inplace=True)

    # paws_df_FILTERED = paws_df[
    #     (paws_df[f"bmp2_temp"] != -999.99) & (paws_df['htu_temp'] != -999.99) & (paws_df['mcp9808'] != -999.99)
    # ]
    # paws_df_FILTERED = paws_df_FILTERED[paws_df_FILTERED['htu_temp'] > -50.0]
    # paws_df_FILTERED = paws_df_FILTERED[paws_df_FILTERED['mcp9808'] > -50.0]    

    #paws_df_FILTERED.to_csv(f'/Users/rzieber/Downloads/{station_directories[i][8:14]}_paws_df_filtered.csv')

    # tsms_df_FILTERED = tsms_df_FILTERED[
    #     (tsms_df_FILTERED['temp'] > -50.0)
    # ]

    # for year_month, paws_group in paws_df_FILTERED.groupby('year_month'):
    #     tsms_group = tsms_df_FILTERED[tsms_df_FILTERED['year_month'] == year_month]
    #     #tsms_group = tsms_df_FILTERED.groupby('year_month')

    #     plt.figure(figsize=(20, 12))

    #     plt.plot(paws_group.index, paws_group["bmp2_temp"], marker='.', markersize=1, label=f"3D PAWS bmp2_temp")
    #     plt.plot(paws_group.index, paws_group["htu_temp"], marker='.', markersize=1, label=f"3D PAWS htu_temp")
    #     plt.plot(paws_group.index, paws_group["mcp9808"], marker='.', markersize=1, label=f"3D PAWS mcp9808")
    #     plt.plot(tsms_group.index, tsms_group['temp'], marker='.', markersize=1, label='TSMS')

    #     plt.title(f'{station_directories[i][8:14]} Temperature Comparison for {year_month}: 3D PAWS versus TSMS')
    #     #plt.title(f'{station_directories[i][8:14]} Temperature Comparison for {year_month}: 3D PAWS')
    #     plt.xlabel('Date')
    #     plt.ylabel('Temperature (˚C)')
    #     plt.xticks(rotation=45)

    #     plt.legend()

    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(data_destination+station_directories[i]+f"temp/filtered_nulls/{station_directories[i][8:14]}_temp_comparison_{year_month}_FILTERED.png")

    #     plt.clf()
    #     plt.close()

