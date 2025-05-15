import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

output = Path(...)
data = Path(...)
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

"""
=============================================================================================================================
Create the 3D PAWS and TSMS dataframes.
=============================================================================================================================    
"""
paws_dfs = []
tsms_dfs = []

for csv in csvs:
    name = csv.stem
    df = pd.read_csv(csv)
    if name.startswith("3DPAWS"): paws_dfs.append(df)
    else: tsms_dfs.append(df)


"""
=============================================================================================================================
Accuracy (systematic errors) -------------------------          
    Bias, Mean Absolute Error, Root Mean Squared Error
=============================================================================================================================    
"""
stats = []  # [TSMS00[TSMS00, var, bias, mae, rmse], TSMS01[TSMS01, var, bias, mae, rmse], ...]
stats.append(['Station', 'Variable', 'Bias', 'Mean Absolute Error', 'Root Mean Squared Error'])

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
                tsms = merged['tsms'].to_numpy()
                paws = merged['paws'].to_numpy()

                s = []
                s.append(f'TSMS0{j}')
                s.append(f'{var}')

                if var in ['wind_dir']: # handle circular discrepancy at 360 -> 0
                    raw_diff = tsms - paws
                    abs_err = np.minimum(np.abs(raw_diff), 360 - np.abs(raw_diff))
                    signed_diff = (raw_diff + 180) % 360 - 180
                    s.append(round(signed_diff.mean(), 2))              # bias
                    s.append(round(abs_err.mean(), 2))                  # mae
                    s.append(round(np.sqrt((abs_err**2).mean()), 2))    # rmse
                elif var in ['tipping']: # exclude dry periods
                    nonzero_mask = ~((tsms == 0) & (paws == 0))
                    tsms_nonzero = tsms[nonzero_mask]
                    paws_nonzero = paws[nonzero_mask]
                    s.append(round((tsms_nonzero - paws_nonzero).mean(), 2))   
                    s.append(round(mean_absolute_error(tsms_nonzero, paws_nonzero), 2))         
                    s.append(round(root_mean_squared_error(tsms_nonzero, paws_nonzero),2))      
                else:
                    s.append(round((tsms - paws).mean(), 2))                   
                    s.append(round(mean_absolute_error(tsms, paws), 2))        
                    s.append(round(root_mean_squared_error(tsms, paws),2))     

                stats.append(s)

df_stats = pd.DataFrame(stats[1:], columns=stats[0])
df_stats.to_csv(output / "accuracy.csv", index=False)


"""
=============================================================================================================================
Precision (random errors) ----------------------
    Standard Deviations, Correlation Coefficients
=============================================================================================================================    
"""
stats = []  # [TSMS00[TSMS00, var, std_paws, std_tsms, r], TSMS01[TSMS01, var, std, r], ...]
stats.append(['Station', 'Variable', '3D-PAWS Standard Deviation', 'TSMS Standard Deviation', 'Pearson Correlation Coefficient'])

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
                tsms = merged['tsms'].to_numpy()
                paws = merged['paws'].to_numpy()

                paws_std, tsms_std, corr_coef = None, None, None
                if var in ['tipping']: # exclude dry periods
                    nonzero_mask = ~((tsms == 0) & (paws == 0))
                    tsms_nonzero = tsms[nonzero_mask]
                    paws_nonzero = paws[nonzero_mask]
                    corr_coef, _ = pearsonr(tsms_nonzero, paws_nonzero)
                    paws_std = np.std(paws_nonzero, ddof=1)
                    tsms_std = np.std(tsms_nonzero, ddof=1)
                else:
                    corr_coef, _ = pearsonr(tsms, paws)
                    paws_std = np.std(paws, ddof=1)
                    tsms_std = np.std(tsms, ddof=1)

                s = [
                    f'TSMS0{j}',
                    f'{var}',
                    round(paws_std, 2),
                    round(tsms_std, 2),
                    round(corr_coef, 2)
                ]
                stats.append(s)

df_stats = pd.DataFrame(stats[1:], columns=stats[0])
df_stats.to_csv(output / "precision.csv", index=False)


"""
=============================================================================================================================
Reliability
=============================================================================================================================    
"""
stats = []  # [TSMS00[TSMS00, var, reliability, inaccuracy], TSMS01[TSMS01, var, reliability, inaccuracy], ...]
stats.append(['Station', 'Variable', 'Reliability', 'Inaccuracy'])

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
                tsms = merged['tsms'].to_numpy()
                paws = merged['paws'].to_numpy()

                abs_diff = None
                if var in ['tipping']: # exclude dry periods
                    nonzero_mask = ~((tsms == 0) & (paws == 0))
                    tsms_nonzero = tsms[nonzero_mask]
                    paws_nonzero = paws[nonzero_mask]
                    abs_diff = np.abs(tsms_nonzero - paws_nonzero)
                else:
                    abs_diff = np.abs(tsms - paws)

                reliability = ((abs_diff <= threshold).sum() / len(abs_diff)) * 100
                inaccuracy = 100 - reliability

                s = [
                    f'TSMS0{j}',
                    f'{var}',
                    round(reliability, 2),
                    round(inaccuracy, 2)
                ]
                stats.append(s)

df_stats = pd.DataFrame(stats[1:], columns=stats[0])
df_stats.to_csv(output / "reliability.csv", index=False)
