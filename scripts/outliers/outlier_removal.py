import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import matplotlib.cm as cm
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import seaborn as sns

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

data_origin = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/output/"
data_destination = r""

station_variables = [ 
    "temp", "humidity", "actual_pressure", "sea_level_pressure", "wind", "total_rainfall"
]
variable_mapper = { # TSMS : 3DPAWS
    "temp":["bmp2_temp", "htu_temp", "mcp9808"],
    "humidity":["bme2_hum", "htu_hum"],
    "actual_pressure":"bmp2_pres",
    "sea_level_pressure":"bmp2_slp",
    "avg_wind_dir":"wind_dir", # both using 1-minute average
    "avg_wind_speed":"wind_speed", # both usng 1-minute average
    "total_rainfall":"tipping"
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
                    os.makedirs(data_destination+station_directories[i]+var+'/filtered_nulls/', exist_ok=True)
            except Exception as e:
                print("Error creating the directory: ", var)
    except Exception as e:
        print("Could not create ", station_directories[i][:len(station_directories[i])-1])
    
# -------------------------------------------------------------------------------------------------------------------------------
        
paws_dfs = [] # stations TSMS00 -> TSMS08
tsms_dfs = [] # Ankara, Konya, Adana

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



# -------------------------------------------------------------------------------------------------------------------------------
"""
Outlier dataframe: 

------------------------------------------------------------
timestamp  |  column name  |  original value |  outlier type
------------------------------------------------------------

"""
# -------------------------------------------------------------------------------------------------------------------------------



        
"""
=============================================================================================================================
TSMS00 Outlier Removal
=============================================================================================================================
"""
paws_df = paws_dfs[0]
outlier_info = [] # [timestamp, column name, original value, outlier type]
#paws_df_FILTERED = paws_df.replace(-999.99, np.nan)




# Temperature -- thresholds from box plots
for temp in variable_mapper["temp"]:
    Q1 = paws_df_FILTERED[f'{temp}'].quantile(0.25)
    Q3 = paws_df_FILTERED[f'{temp}'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    outliers = paws_df_FILTERED[
        (paws_df_FILTERED[f"{temp}"] < lower_bound) | (paws_df_FILTERED[f"{temp}"] > upper_bound)
    ][f"{temp}"]

    outliers_dict[f"{temp}"] = outliers.values

    # INSERT LOGIC FOR MAKING TIME SERIES WITHOUT OUTLIERS HERE!!!

outliers_df = pd.DataFrame(
    dict([(k, pd.Series(v)) for k, v in outliers_dict.items()])
)

outliers_df.to_csv("test_df.csv", index=False)


"""
=============================================================================================================================
TSMS01 Outlier Removal
=============================================================================================================================
"""

"""
=============================================================================================================================
TSMS02 Outlier Removal
=============================================================================================================================
"""

"""
=============================================================================================================================
TSMS03 Outlier Removal
=============================================================================================================================
"""

"""
=============================================================================================================================
TSMS04 Outlier Removal
=============================================================================================================================
"""

"""
=============================================================================================================================
TSMS05 Outlier Removal
=============================================================================================================================
"""

"""
=============================================================================================================================
TSMS06 Outlier Removal
=============================================================================================================================
"""

"""
=============================================================================================================================
TSMS07 Outlier Removal
=============================================================================================================================
"""

"""
=============================================================================================================================
TSMS08 Outlier Removal
=============================================================================================================================
"""
