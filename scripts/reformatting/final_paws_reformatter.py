"""
==========================================================================================
Uses the reformatted 3D-PAWS data from 2022 -> Jan. 2024 and combines the remaining data
from Jan. 2024 - Nov. 2024. 

2022 - Jan. 2024 data origin: SD card
Jan. 2024 - Nov. 2024 data origin: CHORDS
1-minute timestamp offset between these two datasets

SLP approximation consistent across entire collective data period.
==========================================================================================
"""
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

sys.path.append(str(Path(__file__).resolve().parents[2]))
from resources import functions as func


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

# print("Resampling for development purposes < ------------------------------------ disable when done")

# for i in range(len(paws_dfs_partI)):
#     originalI_df = paws_dfs_partI[i]
#     resampledI_df = originalI_df.sample(1000, random_state=42)
#     paws_dfs_partI[i] = resampledI_df

#     originalII_df = paws_dfs_partII[i]
#     resampledII_df = originalII_df.sample(1000, random_state=42)
#     paws_dfs_partII[i] = resampledII_df
## This will not be compatible with data gap infilling. For reformatting only.

print("Building reformatted dataframes for 2022 -> Jan. 2024.")

i = 0
for df in paws_dfs_partI:
    print(f"\tReformatting TSMS0{i}")

    df.columns = df.columns.str.strip()

    df.rename(columns={"mon":"month", "min":"minute"}, inplace=True)

    with warnings.catch_warnings(): # supress warning from printing to the console
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        
        try:
            df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']], errors='coerce')
        except Exception as e:
            print(f"Error converting date for TSMS0{i}: {e}")

    df.drop(   # no reference data for these col's OR no data exists in these columns (e.g. sth) OR scalar (e.g. alt)
        ['int', 'sth_temp', 'sth_hum', 'bmp2_alt', 'vis_light', 'ir_light', 'uv_light', 'year', 'month', 'day', 'hour', 'minute'], 
        axis=1, 
        inplace=True
    )
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)

        df['year_month'] = df['date'].dt.to_period('M')
        df['year_month_day'] = df['date'].dt.to_period('D')
        df['year_month_day_hour'] = df['date'].dt.to_period('H')

    # df.set_index('date', inplace=True)

    paws_dfs_reformatted_partI.append(df)

    i += 1


print("Building reformatted dataframes for Jan. 2024 -> Nov. 2024.")

i = 0
for df in paws_dfs_partII:
    print(f"\tReformatting TSMS0{i}")

    df.columns = df.columns.str.strip()

    if ("htu21d_temp" in df.columns) and ("bmp_slp" not in df.columns):                             # CHORDS variables have so many fun quirks ¯\(°_o)/¯
        df.rename(
            columns={"time":"date", "rain":"tipping", "wind_direction":"wind_dir",
                    "htu21d_temp":"htu_temp", "htu21d_humidity":"htu_hum",
                    "bmp_slp":"bmp2_slp", "bmp_pressure":"bmp2_pres", "bmp_temp":"bmp2_temp"}, 
            inplace=True
        ) 
        df["bmp2_slp"] = func.calc_slp(df, 48) # Adana = 48 m
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
    elif ("sht31d_temp" in df.columns) and ("bmp_slp" not in df.columns):
        
        df.rename(
            columns={"time":"date", "rain":"tipping", "wind_direction":"wind_dir",
                    "sht31d_temp":"htu_temp", "sht31d_humidity":"htu_hum",
                    "bmp_slp":"bmp2_slp", "bmp_pressure":"bmp2_pres", "bmp_temp":"bmp2_temp"}, 
            inplace=True
        )
        df["bmp2_slp"] = func.calc_slp(df, 48) # Adana = 48 m
        print(f"\t\tNo SLP data found for TSMS0{i} -- approximating\n\t\t(see Functions for methodology)")
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
    
    df['bme2_hum'] = np.nan
        
    # no reference data for these col's OR no data exists in these columns (e.g. sth) OR scalar (e.g. alt)
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

        df['year_month'] = df['date'].dt.to_period('M')
        df['year_month_day'] = df['date'].dt.to_period('D')
        df['year_month_day_hour'] = df['date'].dt.to_period('H')
    
    # df.set_index('date', inplace=True)

    paws_dfs_reformatted_partII.append(df)

    i += 1

print("Filling in datagaps.")

i = 0
for i in range(len(paws_dfs_partI)):
    print(f"\tFilling TSMS0{i}")

    num_empty_rows_ptI = paws_dfs_partI[i]['date'].isna().sum()
    num_empty_rows_ptII = paws_dfs_partII[i]['date'].isna().sum()
    if not num_empty_rows_ptI == 0: print("Number of empty rows found part I:", num_empty_rows_ptI)
    if not num_empty_rows_ptII == 0: print("Number of empty rows found part II:", num_empty_rows_ptII)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        gaps_filled_df_ptI = func.fill_empty_rows(paws_dfs_partI[i], timedelta(minutes=1))
        gaps_filled_df_ptII = func.fill_empty_rows(paws_dfs_partII[i], timedelta(minutes=1))

    paws_dfs_partI[i] = gaps_filled_df_ptI
    paws_dfs_partII[i] = gaps_filled_df_ptII

    i += 1
