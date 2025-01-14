"""
==========================================================================================
Uses the reformatted 3D-PAWS data from 2022 -> Jan. 2024 and combines the remaining data
from Jan. 2024 - Nov. 2024. 

To be used in conjunction with the filegen script:
    Step 1. Run final_paws_reformatter.py
    Step 2. Run final_paws_filegen.py
Where the data DESTINATION in step 1 is the data ORIGIN in step 2.


2022 - Jan. 2024 data origin: SD card
Jan. 2024 - Nov. 2024 data origin: CHORDS
1-minute timestamp offset between these two datasets accounted for.

SLP approximation calculated onboard RPi's (instruments 0-5) and calculated here
for instruments 6-8 at Adana, which use Particle dataloggers.
SLP approx. methodology is consistent between RPi's & Particle's (see functions.py)

Sets the "seconds" component of all timestamps to zero. SD card data never contained data
recorded to the second, nor did TSMS.
==========================================================================================
"""
import sys
import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

sys.path.append(str(Path(__file__).resolve().parents[2]))
from resources import functions as func


data_destination = "/Users/rzieber/Documents/3D-PAWS/Turkiye/reformatted/CSV_Format/3DPAWS/full_dataperiod"

data_origin_partI = "/Users/rzieber/Documents/3D-PAWS/Turkiye/reformatted/CSV_Format/3DPAWS/2022-Jan2024/complete_record/"
data_origin_partII = "/Users/rzieber/Documents/3D-PAWS/Turkiye/raw/3DPAWS/Jan-2024_Nov-2024/"

all_files_partI = sorted([file for file in Path(data_origin_partI).rglob('*') if file.is_file() and file.name != ".DS_Store"])
all_files_partII = sorted([file for file in Path(data_origin_partII).rglob('*') if file.is_file() and file.name != ".DS_Store"])

print("Building initial dataframes.")

paws_dfs_partI = [] 
for file in all_files_partI:
    paws_dfs_partI.append(
        pd.read_csv(file, low_memory=False)
    )
paws_dfs_partII = []
for file in all_files_partII:
    paws_dfs_partII.append(
        pd.read_csv(file, low_memory=False)
    )

print("Building reformatted dataframes for 2022 -> Jan. 2024.")

paws_dfs_reformatted_partI = []
i = 0
for df in paws_dfs_partI:
    print(f"\tReformatting TSMS0{i}")

    df.columns = df.columns.str.strip()

    df.rename(columns={"mon":"month", "min":"minute"}, inplace=True)

    with warnings.catch_warnings(): # supress warnings from printing to the console
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        
        try:
            df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']], errors='coerce')
        except Exception as e:
            print(f"Error converting date for TSMS0{i}: {e}")

    df.drop(
        ['year', 'month', 'day', 'hour', 'minute'], 
        axis=1, 
        inplace=True
    )

    if i in [6,7,8]: df['bmp2_slp'] = func.calc_slp(df, 48, 'mcp9808', 'bmp2_pres')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)

        df['year_month'] = df['date'].dt.to_period('M')
        df['year_month_day'] = df['date'].dt.to_period('D')
        df['year_month_day_hour'] = df['date'].dt.to_period('H')
    
    paws_dfs_reformatted_partI.append(df.drop_duplicates(subset='date', keep='first'))

    i += 1

print("Building reformatted dataframes for Jan. 2024 -> Nov. 2024.")

paws_dfs_reformatted_partII = []
i = 0
for df in paws_dfs_partII:
    print(f"\tReformatting TSMS0{i}")

    df.columns = df.columns.str.strip()

    if ("htu21d_temp" in df.columns) and ("bmp_slp" not in df.columns): # CHORDS variables have so many fun quirks ¯\(°_o)/¯
        df.rename(
            columns={"time":"date", "rain":"tipping", "wind_direction":"wind_dir",
                    "htu21d_temp":"htu_temp", "htu21d_humidity":"htu_hum",
                    "bmp_slp":"bmp2_slp", "bmp_pressure":"bmp2_pres", "bmp_temp":"bmp2_temp"}, 
            inplace=True
        ) 
        df["bmp2_slp"] = func.calc_slp(df, 48, "mcp9808", "bmp2_pres")
        print(f"\t\tNo SLP data found for TSMS0{i} -- approximating\n\t\t(see Functions for methodology)")
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
    elif "sth31d_humidity" in df.columns:
        df.rename(
            columns={"time":"date", "rain":"tipping", "wind_direction":"wind_dir",
                    "sth31d_temp":"htu_temp", "sth31d_humidity":"htu_hum",
                    "bmp_slp":"bmp2_slp", "bmp_pressure":"bmp2_pres", "bmp_temp":"bmp2_temp"}, 
            inplace=True
        ) 
    elif ("sht31d_temp" in df.columns) and ("bmp_slp" not in df.columns):
        
        df.rename(
            columns={"time":"date", "rain":"tipping", "wind_direction":"wind_dir",
                    "sht31d_temp":"htu_temp", "sht31d_humidity":"htu_hum",
                    "bmp_slp":"bmp2_slp", "bmp_pressure":"bmp2_pres", "bmp_temp":"bmp2_temp"}, 
            inplace=True
        )
        df["bmp2_slp"] = func.calc_slp(df, 48, 'mcp9808', 'bmp2_pres')
        print(f"\t\tNo SLP data found for TSMS0{i} -- approximating\n\t\t(see Functions for methodology)")
    elif "sht31d_humidity" in df.columns:
        df.rename(
            columns={"time":"date", "rain":"tipping", "wind_direction":"wind_dir",
                    "sht31d_temp":"htu_temp", "sht31d_humidity":"htu_hum",
                    "bmp_slp":"bmp2_slp", "bmp_pressure":"bmp2_pres", "bmp_temp":"bmp2_temp"}, 
            inplace=True
        ) 
    else:
        print(f"Error while renaming files for TSMS0{i}")
        print(f"Headers identified for TSMS{i}:", df.columns)
    
    df['bme2_hum'] = np.nan
    df['sth_temp'] = np.nan
    df['sth_hum'] = np.nan
        
    if all(col in df.columns for col in ['wg', 'wgd', 'wgd_compass_dir']):
        df.drop(   
            ['si1145_vis', 'si1145_ir', 'si1145_uv', 'wind_direction_compass_dir', 
            'wg', 'wgd', 'wgd_compass_dir'], 
            axis=1, 
            inplace=True
        )
    else:
        df.drop( 
            ['si1145_vis', 'si1145_ir', 'si1145_uv', 'wind_direction_compass_dir'], 
            axis=1, 
            inplace=True
        )
    
    if 'bpc' in df.columns: df.drop(['bpc'], axis=1, inplace=True)
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings("ignore", message=".*Converting to PeriodArray.*")

        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df['date'] = df['date'].dt.floor('min') # set "seconds" to zero 

        df['year_month'] = df['date'].dt.to_period('M')
        df['year_month_day'] = df['date'].dt.to_period('D')
        df['year_month_day_hour'] = df['date'].dt.to_period('H')

    paws_dfs_reformatted_partII.append(df.drop_duplicates(subset='date', keep='first'))

    i += 1

print("Filling in datagaps.")

dfs_gaps_filled_partI = []
dfs_gaps_filled_partII = []

for i in range(len(paws_dfs_reformatted_partI)):
    print(f"\tFilling TSMS0{i}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        gaps_filled_df_ptI = func.fill_empty_rows(paws_dfs_reformatted_partI[i], timedelta(minutes=1))
        gaps_filled_df_ptII = func.fill_empty_rows(paws_dfs_reformatted_partII[i], timedelta(minutes=1))

    dfs_gaps_filled_partI.append(gaps_filled_df_ptI)
    dfs_gaps_filled_partII.append(gaps_filled_df_ptII)
    
print("Merging 1st and 2nd data periods.")

complete_records = []
for i in range(len(dfs_gaps_filled_partI)):
    print(f"\tMerging TSMS0{i}")

    df_partI = dfs_gaps_filled_partI[i]
    df_partII = dfs_gaps_filled_partII[i]

    # account for sd card & chords 1-minute offset --> sd card timestamps 1 minute AHEAD of chords
    with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)

            last_timestamp = df_partI['date'].iloc[-1]
            last_timestamp += timedelta(minutes=2)
            df_partII = df_partII[df_partII['date'] >= last_timestamp]
            df_partII['date'] = df_partII['date'] - timedelta(minutes=1)

    complete_record = pd.concat([df_partI, df_partII], axis=0, join='outer')

    complete_records.append(complete_record)

print("Creating final CSV's")

i = 0
for df in complete_records:
    print(f"\tConverting TSMS0{i}")

    df['sth_hum'] = np.nan

    file_path = os.path.join(data_destination, f"station_TSMS0{i}", f"TSMS0{i}_CompleteRecord.csv")
    df.to_csv(file_path)

    i += 1
