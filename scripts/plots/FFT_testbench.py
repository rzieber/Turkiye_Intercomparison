import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Define low-pass filter
def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


df_paws = pd.read_csv("/Users/rzieber/Documents/3D-PAWS/Turkiye/reformatted/CSV_Format/analysis/station_TSMS00/TSMS00_CompleteRecord_FINAL.csv",
                      low_memory=False,
                      header=0)
df_tsms = pd.read_csv("/Users/rzieber/Documents/3D-PAWS/Turkiye/reformatted/CSV_Format/analysis/station_TSMS00/Ankara_CompleteRecord_FINAL.csv",
                      low_memory=False,
                      header=0)
outliers = pd.DataFrame()
outliers_to_add = []

df_paws['date'] = pd.to_datetime(df_paws['date'])
# df_paws.set_index('date', inplace=True)

# Apply filter
filtered_data = low_pass_filter(df_paws['htu_temp'], cutoff=0.0160, fs=1/60)
outliers = df_paws['htu_temp'] - filtered_data

for dp in filtered_data:
    print(dp)

#filtered_data.to_csv('/Users/rzieber/Downloads/filtered_data_TSMS00.csv')
# np.savetxt('/Users/rzieber/Downloads/filtered_data_TSMS00.csv', filtered_data, delimiter=',', fmt='%d', header='Date, Temperature')
# outliers.to_csv('/Users/rzieber/Downloads/outliers_from_FFT.csv')

# -------------------------------------------------------------------
# first_timestamp = df_paws.index[0]
# df_paws_sliced = df_paws[first_timestamp:'2022-08-19 00:00:00']

# df_paws_sliced.reset_index('date')

# i = 0
# this_timestamp = df_paws['date'].iloc[i]
# #this_timestamp = df_paws.index[i]
# while i < len(df_paws) -2:
#     next_datapoint = i+1

#     # skip over empty cells in the dataframe
#     while (df_paws.loc[next_datapoint, 'htu_temp'] == np.nan) and (next_datapoint != len(df_paws)-1): 
#         next_datapoint += 1
    
#     next_timestamp = df_paws['date'].iloc[next_datapoint] # get the timestamp associated with next_datapoint

#     tsms_reference_temp = df_tsms.loc[(df_tsms.index == next_timestamp), 'temperature'].values # grab the temperature value from reference data at that timestamp

#     diff_from_neighbor = abs(df_paws.loc[i, 'htu_temp'] - df_paws.loc[next_datapoint, 'htu_temp']) # get the abs differences from current temp to neighboring temp in dataframe
#     diff_from_tsms = abs(df_paws.loc[next_datapoint, 'htu_temp'] - tsms_reference_temp) # get the abs difference from the neighboring temp to the temp in reference data at the same timestamp
    
#     threshold = 1.7

#     if diff_from_neighbor > threshold: # if threshold value exceeded
#         outliers_to_add.append({ # record the bad datapoint
#             'date':df_paws.loc[next_datapoint, 'date'],
#             'column_name': 'htu_temp',
#             'original_value':df_paws.loc[next_datapoint, 'htu_temp'],
#             'outlier_type': 'htu_trend_switching'
#         })

#         df_paws.loc[next_datapoint, 'htu_temp'] = np.nan # replace bad datapoint in original dataframe
#     else:
#         i = next_datapoint # if the difference doesn't exceed threshold, set i to the next non-empty datapoint
# -------------------------------------------------------------------

# outliers = pd.concat([outliers, outliers_to_add], ignore_index=True)

# outliers.to_csv("/Users/rzieber/Downloads/outliers_removed.csv")
# df_paws.to_csv("/Users/rzieber/Downloads/paws_cleaned.csv")
