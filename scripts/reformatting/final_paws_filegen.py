import os
import pandas as pd
import numpy as np
from pathlib import Path


data_origin = "/Users/rzieber/Documents/3D-PAWS/Turkiye/reformatted/CSV_Format/3DPAWS/Aug2022-Nov2024/FINAL/"
all_files = sorted([file for file in Path(data_origin).rglob('*') if file.is_file() and file.name != ".DS_Store" and "_FINAL.csv" not in file.name])

dfs = []

for file in all_files:
    dfs.append(pd.read_csv(file, low_memory=False))

print("Creating CSV's for 3D-PAWS full data period.")

for i in range(len(dfs)):
    print(f"\tGenerating TSMS0{i}")

    df = dfs[i]
    file = all_files[i]
    filepath = str(file)[:len(str(file))-25]

    for year_month, paws_grouped in df.groupby('year_month'):
        print(f"\t\t{year_month}")

        paws_grouped.drop(columns=['Unnamed: 0','year_month', 'year_month_day', 'year_month_day_hour'], inplace=True)

        if i in [0,1,2]:    # Ankara
            paws_grouped['bmp2_alt'] = 935
        elif i in [4,5,6]:  # Konya
            paws_grouped['bmp2_alt'] = 1016
        else:               # Adana
            paws_grouped['bmp2_alt'] = 48
        
        paws_grouped['sth_temp'] = np.nan
        paws_grouped['sth_hum'] = np.nan
        paws_grouped['int'] = 900

        paws_grouped = paws_grouped[['date', 'int', 'bmp2_temp', 'htu_temp', 'mcp9808', 'sth_temp', 'sth_hum', 'htu_hum', \
                'bme2_hum', 'bmp2_pres', 'bmp2_slp', 'bmp2_alt', 'tipping', 'wind_speed', 'wind_dir']]

        paws_grouped.set_index('date', inplace=True)

        paws_grouped.to_csv(filepath+f"/monthly/TSMS0{i}_{year_month}_FINAL.csv") 

    for year_month_day, paws_grouped in df.groupby('year_month_day'):
        print(f"\t\t{year_month_day}")

        paws_grouped.drop(columns=['Unnamed: 0','year_month', 'year_month_day', 'year_month_day_hour'], inplace=True)

        if i in [0,1,2]:    # Ankara
            paws_grouped['bmp2_alt'] = 935
        elif i in [4,5,6]:  # Konya
            paws_grouped['bmp2_alt'] = 1016
        else:               # Adana
            paws_grouped['bmp2_alt'] = 48
        
        paws_grouped['sth_temp'] = np.nan
        paws_grouped['sth_hum'] = np.nan
        paws_grouped['int'] = 900

        paws_grouped = paws_grouped[['date', 'int', 'bmp2_temp', 'htu_temp', 'mcp9808', 'sth_temp', 'sth_hum', 'htu_hum', \
                'bme2_hum', 'bmp2_pres', 'bmp2_slp', 'bmp2_alt', 'tipping', 'wind_speed', 'wind_dir']]

        paws_grouped.set_index('date', inplace=True)

        paws_grouped.to_csv(filepath+f"/daily/TSMS0{i}_{year_month_day}_FINAL.csv")

    
    df.drop(columns=['Unnamed: 0','year_month', 'year_month_day', 'year_month_day_hour'], inplace=True)

    if i in [0,1,2]:    # Ankara
        df['bmp2_alt'] = 935
    elif i in [4,5,6]:  # Konya
        df['bmp2_alt'] = 1016
    else:               # Adana
        df['bmp2_alt'] = 48
    
    df['sth_temp'] = np.nan
    df['sth_hum'] = np.nan
    df['int'] = 900

    df = df[['date', 'int', 'bmp2_temp', 'htu_temp', 'mcp9808', 'sth_temp', 'sth_hum', 'htu_hum', \
            'bme2_hum', 'bmp2_pres', 'bmp2_slp', 'bmp2_alt', 'tipping', 'wind_speed', 'wind_dir']]

    df.set_index('date', inplace=True)

    df.to_csv(filepath+f"/complete/TSMS0{i}_CompleteRecord_FINAL.csv")

    os.remove(str(os.path.join(data_origin, f"station_TSMS0{i}", f"TSMS0{i}_CompleteRecord.csv")))
