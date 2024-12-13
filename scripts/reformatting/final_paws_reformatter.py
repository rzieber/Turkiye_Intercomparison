"""
==========================================================================================
Uses the reformatted 3D-PAWS data from 2022 -> Jan. 2024 and combines the remaining data
from Jan. 2024 - Nov. 2024. 

2022 - Jan. 2024 data origin: SD card
Jan. 2024 - Nov. 2024 data origin: CHORDS

SLP approximation consistent across entire collective data period.
==========================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings


# Functions ===============================================================================

"""
Calculates the sea level pressure using observed station pressure and the station elevation (in meters).
Returns a Series to be added as a column to the dataframe.
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
def calc_slp(df:pd.DataFrame, alt:int) -> pd.Series:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The calc_slp() function expects 'df' parameter to be of type <DataFrame>, passed: {type(df)}")
    if not isinstance(alt, int):
        raise TypeError(f"The calc_slp() function expects 'alt' parameter to be of type <int>, passed: {type(alt)}")
    
    return df["bmp2_pres"]*(293/(293-(alt*0.0065)))
    
# =========================================================================================

data_origin_partI = "/Users/rzieber/Documents/3D-PAWS/Turkiye/reformatted/CSV_Format/3DPAWS/complete_record/"
data_origin_partII = "/Users/rzieber/Documents/3D-PAWS/Turkiye/raw/3DPAWS/Jan-2024_Nov-2024/"
all_files_partI = sorted([file for file in Path(data_origin_partI).rglob('*') if file.is_file() and file.name != ".DS_Store"])
all_files_partII = sorted([file for file in Path(data_origin_partII).rglob('*') if file.is_file() and file.name != ".DS_Store"])

paws_dfs_partI = []
paws_dfs_reformatted_partI = []
paws_dfs_partII = []
paws_dfs_reformatted_partII = []

print("Building initial dataframes.")

for file in all_files_partI:
    paws_dfs_partI.append(
        pd.read_csv(file, low_memory=False)
    )
for file in all_files_partII:
    paws_dfs_partII.append(
        pd.read_csv(file, low_memory=False)
    )

print("Building reformatted dataframes for 2022 -> Jan. 2024.")

for df in paws_dfs_partI:
    df.columns = df.columns.str.strip()

    df.rename(columns={"mon":"month", "min":"minute"}, inplace=True)

    with warnings.catch_warnings(): # supress warning from printing to the console
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        
        df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])

    df.drop(   # no reference data for these col's OR no data exists in these columns (e.g. sth) OR scalar (e.g. alt)
        ['int', 'sth_temp', 'sth_hum', 'bmp2_alt', 'vis_light', 'ir_light', 'uv_light', 'year', 'month', 'day', 'hour', 'minute'], 
        axis=1, 
        inplace=True
    )

    df = df[['date', 'bmp2_temp', 'htu_temp', 'mcp9808', 'bme2_hum', 'htu_hum', 'bmp2_pres',
                                                'bmp2_slp', 'wind_dir', 'wind_speed', "tipping"]]
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)

        df['year_month'] = df['date'].dt.to_period('M')
        df['year_month_day'] = df['date'].dt.to_period('D')
        df['year_month_day_hour'] = df['date'].dt.to_period('H')

    
    df.set_index('date', inplace=True)

    paws_dfs_reformatted_partI.append(df)


print("Building reformatted dataframes for Jan. 2024 -> Nov. 2024.")

i = 0
for df in paws_dfs_partII:
    print(f"\tReformatting TSMS0{i}")

    df.columns = df.columns.str.strip()

    if ("htu21d_temp" in df.columns) and ("bmp_slp" not in df.columns):
        print(f"No SLP data found for TSMS0{i} -- approximating (see Functions for methodology).")
        df.rename(
            columns={"time":"date", "rain":"tipping", "wind_direction":"wind_dir",
                    "htu21d_temp":"htu_temp", "htu21d_humidity":"htu_hum",
                    "bmp_slp":"bmp2_slp", "bmp_pressure":"bmp2_pres", "bmp_temp":"bmp2_temp"}, 
            inplace=True
        ) 
        df["bmp2_slp"] = calc_slp(df, 48) # Adana
    elif "htu21d_temp" in df.columns:
        df.rename(
            columns={"time":"date", "rain":"tipping", "wind_direction":"wind_dir",
                    "htu21d_temp":"htu_temp", "htu21d_humidity":"htu_hum",
                    "bmp_slp":"bmp2_slp", "bmp_pressure":"bmp2_pres", "bmp_temp":"bmp2_temp"}, 
            inplace=True
        )
    elif "sth31d_temp" in df.columns:
        df.rename(
            columns={"time":"date", "rain":"tipping", "wind_direction":"wind_dir",
                    "sth31d_temp":"htu_temp", "sth31d_humidity":"htu_hum",
                    "bmp_slp":"bmp2_slp", "bmp_pressure":"bmp2_pres", "bmp_temp":"bmp2_temp"}, 
            inplace=True
        ) 
    elif ("sht31d_temp" in df.columns) and ("bmp_slp" not in df.columns):
        print(f"No SLP data found for TSMS0{i} -- approximating (see Functions for methodology).")
        df.rename(
            columns={"time":"date", "rain":"tipping", "wind_direction":"wind_dir",
                    "sht31d_temp":"htu_temp", "sht31d_humidity":"htu_hum",
                    "bmp_slp":"bmp2_slp", "bmp_pressure":"bmp2_pres", "bmp_temp":"bmp2_temp"}, 
            inplace=True
        )
        df["bmp2_slp"] = calc_slp(df, 48) # Adana
    elif "sht31d_temp" in df.columns:
        df.rename(
            columns={"time":"date", "rain":"tipping", "wind_direction":"wind_dir",
                    "sht31d_temp":"htu_temp", "sht31d_humidity":"htu_hum",
                    "bmp_slp":"bmp2_slp", "bmp_pressure":"bmp2_pres", "bmp_temp":"bmp2_temp"}, 
            inplace=True
        ) 
    else:
        print(f"Error while renaming files for TSMS0{i}")
        print(f"Headers identified for TSMS{i}:", df.columns)
        
    if all(col in df.columns for col in ['wg', 'wgd', 'bpc', 'wgd_compass_dir']):
        df.drop(   # no reference data for these col's OR no data exists in these columns (e.g. sth) OR scalar (e.g. alt)
            ['si1145_vis', 'si1145_ir', 'si1145_uv', 'wind_direction_compass_dir', 
             'wg', 'wgd', 'wgd_compass_dir', 'bpc'], 
            axis=1, 
            inplace=True
        )
    else:
        df.drop(   # no reference data for these col's OR no data exists in these columns (e.g. sth) OR scalar (e.g. alt)
            ['si1145_vis', 'si1145_ir', 'si1145_uv', 'wind_direction_compass_dir'], 
            axis=1, 
            inplace=True
        )

    df = df[['date', 'bmp2_temp', 'htu_temp', 'mcp9808', 'htu_hum', 'bmp2_pres', # 'bme2_hum',
                                                'bmp2_slp', 'wind_dir', 'wind_speed', "tipping"]]
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings("ignore", message=".*Converting to PeriodArray.*")

        df['date'] = pd.to_datetime(df['date'])

        df['year_month'] = df['date'].dt.to_period('M')
        df['year_month_day'] = df['date'].dt.to_period('D')
        df['year_month_day_hour'] = df['date'].dt.to_period('H')
    
    df.set_index('date', inplace=True)

    df.to_csv(f"/Users/rzieber/Downloads/TSMS0{i}_latest_data.csv")

    paws_dfs_reformatted_partII.append(df)

    i += 1

