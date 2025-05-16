"""
NOTE: When calculating these statistics, 2 things needed to be taken into account: 
        - HTU bit-switching noise (timeframes were excluded from analysis for afflicted stations)
        - rainfall can only consider precip events in calculations, all pairs of 0.0's excluded
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

output = Path(r"C:\\Users\\Becky\\Documents\\UCAR_ImportantStuff\\Turkiye\\data\\stats\\")
data = Path(r"C:\\Users\\Becky\\Documents\\UCAR_ImportantStuff\\Turkiye\\data\\final\\csvs\\")
csvs = data.glob("*.csv")


variable_mapper = { # TSMS : 3DPAWS
    "temperature":["bmp2_temp", "htu_temp", "mcp9808"],
    "humidity":["bme2_hum", "htu_hum"],
    "actual_pressure":["bmp2_pres"],
    "sea_level_pressure":["bmp2_slp"],
    "avg_wind_dir":["wind_dir"],
    "avg_wind_speed":["wind_speed"], 
    "total_rainfall":["tipping"]
}
# indexed by variable_mapper keys
wmo_thresholds = [0.2, 2, 0.1, 0.1, 0.3, 3, 0.2] 

instrument_site_mapper = {
    0:[0,1,2], 1:[3,4,5], 2:[6,7,8]
}
sites = ["Ankara", "Konya", "Adana"]

# exclude HTU noise
exclude_htu_noise = [   # Exclude HTU bit-switching from analysis for temperature & rh
    [ 
        "2022-08", "2022-12", "2023-01", "2023-06", "2023-07", "2023-08", "2023-09", 
        "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04", 
        "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11"
    ],
    [
        "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04", "2023-05",
        "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2024-01",
        "2024-03", "2024-04", "2024-05", "2024-11"
    ],
    [
        "2022-09", "2022-11", "2022-12", "2023-01", "2023-04", "2023-05", "2023-06",
        "2023-08", "2023-09", "2023-10", "2023-11"
    ],
    [
        "2023-02", "2023-03", "2023-04"
    ],
    [
        "2024-11"
    ]
]


"""
=============================================================================================================================
Create the 3D PAWS and TSMS dataframes.
Optional -- truncate to match TSMS's 6 month independent study
=============================================================================================================================    
"""
paws_dfs = []
tsms_dfs = []

for csv in csvs:
    name = csv.stem
    df = pd.read_csv(csv, parse_dates=['date'])
    df.set_index('date')

    # full data period
    if name.startswith("3DPAWS"): paws_dfs.append(df)
    else: tsms_dfs.append(df)

    # 6 month data period -- TSMS's independent study
    # data_periods = {
    #     "TSMS00": ["2022-08-23 08:40", "2023-02-23 07:54"],
    #     "TSMS01": ["2022-08-23 08:42", "2023-02-24 12:38"],
    #     "TSMS02": ["2022-08-23 08:44", "2023-02-24 12:42"],
    #     "TSMS03": ["2022-08-22 08:19", "2023-02-24 12:46"],
    #     "TSMS04": ["2022-08-22 08:48", "2023-02-24 13:55"],
    #     "TSMS05": ["2022-08-22 09:35", "2023-02-24 13:57"],
    #     "TSMS06": ["2022-10-26 20:55", "2023-02-24 13:47"],
    #     "TSMS07": ["2022-10-27 17:37", "2023-02-24 13:49"],
    #     "TSMS08": ["2022-10-31 21:58", "2023-02-24 14:02"]
    # }
    # for key in data_periods.keys():
    #     if key in name:
    #         paws_df = df.truncate


"""
=============================================================================================================================
Accuracy (systematic errors) -------------------------          
    Bias, Mean Absolute Error, Root Mean Squared Error
=============================================================================================================================    
"""
accuracy = []  # [TSMS00[TSMS00, var, bias, mae, rmse], TSMS01[TSMS01, var, bias, mae, rmse], ...]
accuracy.append(['Station', 'Variable', 'Bias', 'Mean Absolute Error', 'Root Mean Squared Error'])

for i in range(3):
    print(sites[i])
    tsms_df = tsms_dfs[i]

    for j in instrument_site_mapper[i]:
        print(f"\tTSMS0{j}")
        paws_df = paws_dfs[j]

        for variable in variable_mapper.keys():
            print(f'\t\t{variable}')

            for var in variable_mapper[variable]:
                if var in ['bme2_hum']: continue

                merged = pd.merge(
                    tsms_df[['date', variable]].rename(columns={variable:'tsms'}), 
                    paws_df[['date', var]].rename(columns={var:'paws'}), 
                    on='date', 
                    how='inner'
                )
                merged = merged.dropna(subset=['tsms', 'paws'])

                # exclude HTU noise from calculations
                # if (j in [0,1,2,3,4]) and (var in ['htu_temp', 'htu_hum']):
                #     merged['year_month'] = merged['date'].dt.strftime('%Y-%m')
                #     months_to_exclude = exclude_htu_noise[j]
                #     mask = ~merged['year_month'].isin(months_to_exclude)
                #     merged_2 = merged[mask]
                #     merged = merged_2

                tsms = merged['tsms'].to_numpy()
                paws = merged['paws'].to_numpy()

                # compare daily averages 
                daily_avg = merged.resample('D', on='date').mean().dropna(subset=['tsms', 'paws'])
                tsms_daily = daily_avg['tsms'].to_numpy()
                paws_daily = daily_avg['paws'].to_numpy()

                s = []
                s.append(f'TSMS0{j}')
                s.append(f'{var}')

                if var in ['wind_dir']: # handle circular discrepancy at 360 -> 0
                    raw_diff = tsms_daily - paws_daily
                    # raw_diff = tsms - paws
                    abs_err = np.minimum(np.abs(raw_diff), 360 - np.abs(raw_diff))
                    signed_diff = (raw_diff + 180) % 360 - 180
                    s.append(round(signed_diff.mean(), 2))              # bias
                    s.append(round(abs_err.mean(), 2))                  # mae
                    s.append(round(np.sqrt((abs_err**2).mean()), 2))    # rmse
                elif var in ['tipping']: # exclude dry periods
                    daily_total = merged.resample('D', on='date').sum().dropna(subset=['tsms', 'paws'])
                    tsms_daily = daily_total['tsms'].to_numpy()
                    paws_daily = daily_total['paws'].to_numpy()
                    nonzero_mask = ~((tsms_daily == 0) & (paws_daily == 0))
                    tsms_nonzero = tsms_daily[nonzero_mask]
                    paws_nonzero = paws_daily[nonzero_mask]
                    # nonzero_mask = ~((tsms == 0) & (paws == 0))
                    # tsms_nonzero = tsms[nonzero_mask]
                    # paws_nonzero = paws[nonzero_mask]
                    s.append(round((paws_nonzero - tsms_nonzero).mean(), 2))   
                    s.append(round(mean_absolute_error(tsms_nonzero, paws_nonzero), 2))         
                    s.append(round(root_mean_squared_error(tsms_nonzero, paws_nonzero), 2))  
                else:
                    s.append(round((tsms_daily - paws_daily).mean(), 2))                   
                    s.append(round(mean_absolute_error(tsms_daily, paws_daily), 2))        
                    s.append(round(root_mean_squared_error(tsms_daily, paws_daily), 2))     
                    # s.append(round((paws - tsms).mean(), 2))                   
                    # s.append(round(mean_absolute_error(tsms, paws), 2))        
                    # s.append(round(root_mean_squared_error(tsms, paws), 2))   

                accuracy.append(s)

df_accuracy = pd.DataFrame(accuracy[1:], columns=accuracy[0])

"""
=============================================================================================================================
Precision (random errors) ----------------------
    Standard Deviations, Correlation Coefficients
=============================================================================================================================    
"""
precision = []  # [TSMS00[TSMS00, var, std_paws, std_tsms, r], TSMS01[TSMS01, var, std, r], ...]
precision.append(['Station', 'Variable', '3D-PAWS Standard Deviation', 'TSMS Standard Deviation', 'Pearson Correlation Coefficient'])

for i in range(3):
    print(sites[i])
    tsms_df = tsms_dfs[i]

    for j in instrument_site_mapper[i]:
        print(f"\tTSMS0{j}")
        paws_df = paws_dfs[j]

        for variable in variable_mapper.keys():
            print(f'\t\t{variable}')

            for var in variable_mapper[variable]:
                if var in ['bme2_hum']: continue

                merged = pd.merge(
                    tsms_df[['date', variable]].rename(columns={variable:'tsms'}), 
                    paws_df[['date', var]].rename(columns={var:'paws'}), 
                    on='date', 
                    how='inner'
                )
                merged = merged.dropna(subset=['tsms', 'paws'])

                # exclude HTU noise from calculations
                # if (j in [0,1,2,3,4]) and (var in ['htu_temp', 'htu_hum']):
                #     merged['year_month'] = merged['date'].dt.strftime('%Y-%m')
                #     months_to_exclude = exclude_htu_noise[j]
                #     mask = ~merged['year_month'].isin(months_to_exclude)
                #     merged_2 = merged[mask]
                #     merged = merged_2

                tsms = merged['tsms'].to_numpy()
                paws = merged['paws'].to_numpy()

                # compare daily averages 
                daily_avg = merged.resample('D', on='date').mean().dropna(subset=['tsms', 'paws'])
                tsms_daily = daily_avg['tsms'].to_numpy()
                paws_daily = daily_avg['paws'].to_numpy()

                paws_std, tsms_std, corr_coef = None, None, None
                if var in ['tipping']: # exclude dry periods
                    daily_total = merged.resample('D', on='date').sum().dropna(subset=['tsms', 'paws'])
                    tsms_daily = daily_total['tsms'].to_numpy()
                    paws_daily = daily_total['paws'].to_numpy()
                    nonzero_mask = ~((tsms_daily == 0) & (paws_daily == 0))
                    tsms_nonzero = tsms_daily[nonzero_mask]
                    paws_nonzero = paws_daily[nonzero_mask]
                    # nonzero_mask = ~((tsms == 0) & (paws == 0))
                    # tsms_nonzero = tsms[nonzero_mask]
                    # paws_nonzero = paws[nonzero_mask]

                    corr_coef, _ = pearsonr(tsms_nonzero, paws_nonzero)
                    paws_std = np.std(paws_nonzero, ddof=1)
                    tsms_std = np.std(tsms_nonzero, ddof=1)
                else:
                    corr_coef, _ = pearsonr(tsms_daily, paws_daily)
                    paws_std = np.std(paws_daily, ddof=1)
                    tsms_std = np.std(tsms_daily, ddof=1)
                    # corr_coef, _ = pearsonr(tsms, paws)
                    # paws_std = np.std(paws, ddof=1)
                    # tsms_std = np.std(tsms, ddof=1)

                s = [
                    f'TSMS0{j}',
                    f'{var}',
                    round(paws_std, 2),
                    round(tsms_std, 2),
                    round(corr_coef, 2)
                ]
                precision.append(s)

df_precision = pd.DataFrame(precision[1:], columns=precision[0])

"""
=============================================================================================================================
Reliability
=============================================================================================================================    
"""
reliable = []  # [TSMS00[TSMS00, var, reliability, inaccuracy], TSMS01[TSMS01, var, reliability, inaccuracy], ...]
reliable.append(['Station', 'Variable', 'Reliability', 'Inaccuracy'])

for i in range(3):
    print(sites[i])
    tsms_df = tsms_dfs[i]

    for j in instrument_site_mapper[i]:
        print(f"\tTSMS0{j}")
        paws_df = paws_dfs[j]

        for variable, threshold in zip(variable_mapper.keys(), wmo_thresholds):
            print(f'\t\t{variable}')

            for var in variable_mapper[variable]:
                if var in ['bme2_hum']: continue

                merged = pd.merge(
                    tsms_df[['date', variable]].rename(columns={variable:'tsms'}), 
                    paws_df[['date', var]].rename(columns={var:'paws'}), 
                    on='date', 
                    how='inner'
                )
                merged = merged.dropna(subset=['tsms', 'paws'])

                # exclude HTU noise from calculations
                # if (j in [0,1,2,3,4]) and (var in ['htu_temp', 'htu_hum']):
                #     merged['year_month'] = merged['date'].dt.strftime('%Y-%m')
                #     months_to_exclude = exclude_htu_noise[j]
                #     mask = ~merged['year_month'].isin(months_to_exclude)
                #     merged_2 = merged[mask]
                #     merged = merged_2

                tsms = merged['tsms'].to_numpy()
                paws = merged['paws'].to_numpy()

                # compare daily average 
                daily_avg = merged.resample('D', on='date').mean().dropna(subset=['tsms', 'paws'])
                tsms_daily = daily_avg['tsms'].to_numpy()
                paws_daily = daily_avg['paws'].to_numpy()

                abs_diff = None
                if var in ['tipping']: # exclude dry periods
                    daily_total = merged.resample('D', on='date').sum().dropna(subset=['tsms', 'paws'])
                    tsms_daily = daily_total['tsms'].to_numpy()
                    paws_daily = daily_total['paws'].to_numpy()
                    nonzero_mask = ~((tsms_daily == 0) & (paws_daily == 0))
                    tsms_nonzero = tsms_daily[nonzero_mask]
                    paws_nonzero = paws_daily[nonzero_mask]
                    # nonzero_mask = ~((tsms == 0) & (paws == 0))
                    # tsms_nonzero = tsms[nonzero_mask]
                    # paws_nonzero = paws[nonzero_mask]

                    abs_diff = np.abs(tsms_nonzero - paws_nonzero)
                else:
                    # abs_diff = np.abs(tsms - paws)
                    abs_diff = np.abs(tsms_daily - paws_daily)

                reliability = ((abs_diff <= threshold).sum() / len(abs_diff)) * 100
                inaccuracy = 100 - reliability

                s = [
                    f'TSMS0{j}',
                    f'{var}',
                    round(reliability, 2),
                    round(inaccuracy, 2)
                ]
                reliable.append(s)

df_reliable = pd.DataFrame(reliable[1:], columns=reliable[0])


"""
=============================================================================================================================
Create comprehensive csv
=============================================================================================================================    
"""
key_cols = ['Station', 'Variable']

# For each DataFrame, select only key columns + non-key columns (excluding duplicates)
def select_merge_cols(df):
    return df[key_cols + [col for col in df.columns if col not in key_cols]]

dfs = [df_accuracy, df_precision, df_reliable]
dfs_selected = [select_merge_cols(df) for df in dfs]

df_merged = reduce(
    lambda left, right: pd.merge(left, right, on=key_cols, how='outer'),
    dfs_selected
)

df_merged.to_csv(output / 'all.csv', index=False)
