"""
NOTE: This used to be in outlier-removal.py.
        Failed signal processing logic. The naive approach was sort of working though...
"""

# THIS GOES RIGHT BENEATH THE STATISTICAL OUTLIER REMOVAL FOR 3D-PAWS IN OUTLIER-REMOVAL.PY ########################################

# # Abs Diff -------------- FOR THE HTU ONLY!!!!!!!!
            # if var != "htu_hum": continue;

            # threshold = 1.2
            # last_non_null = None
            # diff_outliers = []

            # i = 0
            # while i < len(paws_df_FILTERED) - 2:
            #     if pd.isna(last_non_null): print("Error with ", last_non_null, ". Current index: ", i)
                
            #     if pd.notna(paws_df_FILTERED.loc[i, var]): 
            #         last_non_null = paws_df_FILTERED.loc[i, var]
            #     if pd.isna(paws_df_FILTERED.loc[i+1, var]): 
            #         i += 1
            #         continue

            #     if pd.isna(last_non_null) and i == 0: continue

            #     diff = abs(paws_df_FILTERED.loc[i+1, var] - last_non_null)
                
            #     if diff is None or last_non_null is None: 
            #         print("Diff: ", diff, "\t\tType: ", type(diff))
            #         print("Last non null: ", last_non_null, "\t\tType: ", type(last_non_null))
            #         print(i)
            #         i += 1
            #         continue

            #     if diff > threshold: 
            #         diff_outliers.append({
            #             'date': paws_df_FILTERED.loc[i+1, 'date'],
            #             'column_name': var,
            #             'original_value': paws_df_FILTERED.loc[i+1, var],
            #             'outlier_type': 'diff contextual'
            #         })

            #         paws_df_FILTERED.loc[i+1, var] = np.nan

            #     i += 1
            
            # diff_outliers_df = pd.DataFrame(diff_outliers)
            # ----------------------- 

##################################################################################################################################



"""
=============================================================================================================================
Phase 5: Filter out HTU bit-switching. 3D-PAWS only.
4/28/25 -- Analysis will have to be restricted to instruments 5-8, can't get signal processing to work in time.
            This wasn't working as well as the naive approach was.
=============================================================================================================================
"""
#     print(f"Phase 5: Filtering out HTU trend-switching.")

#     paws_df_FILTERED.reset_index(drop=True, inplace=True)   # Reset the index to ensure it is a simple range

#     #existing_nulls = paws_df_FILTERED['htu_temp'].isnull()

#     paws_df_FILTERED['htu_temp'] = pd.to_numeric(paws_df_FILTERED['htu_temp'], errors='coerce')


# # -------------------------------------------------------------------------------------------------------------------------------
#     plt.figure(figsize=(20, 12))

#     plt.plot(paws_df_FILTERED['date'], paws_df_FILTERED['htu_temp'], marker='.', markersize=1, label=f"3D PAWS htu_temp")

#     plt.title(f'{station_directories[i][8:14]} Temperature: 3D PAWS htu_temp [BEFORE CLEANING]')
#     plt.xlabel('Date')
#     plt.ylabel('Temperature (˚C)')
#     plt.xticks(rotation=45)

#     plt.legend()

#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(data_destination+station_directories[i]+"\\temperature\\TSMS00_htu-temp_BEFORE.png")

#     plt.clf()
#     plt.close()
# # -------------------------------------------------------------------------------------------------------------------------------

#     #np.savetxt(data_destination+'htu_temp_array.txt', paws_df_FILTERED['htu_temp'])

#     outliers_to_add = []

#     j = 0
#     ground_truth = paws_df_FILTERED['htu_temp'].iloc[j]

#     if pd.isna(ground_truth):
#         while pd.isna(ground_truth):
#             j += 1
#             ground_truth = paws_df_FILTERED['htu_temp'].iloc[j]
#             print("Inside loop:", ground_truth)

#     print(ground_truth, "\tIndex number:", j) # MUST be initialized to a valid temperature reading

#     while j < len(paws_df_FILTERED)-1:
#         j += 1

#         neighbor = paws_df_FILTERED['htu_temp'].iloc[j]

#         if pd.isna(neighbor): continue 

#         if abs(ground_truth - neighbor) > 1.2:
#             # note the original value in neighbor in the outlier table
#             outliers_to_add.append({
#                 'date':paws_df_FILTERED['date'].iloc[j],
#                 'column_name':'htu_temp',
#                 'original_value':neighbor,
#                 'outlier_type':outlier_reasons[4]
#             })
#             # replace the original value in neighbor with np.nan
#             paws_df_FILTERED.loc[j, 'htu_temp'] = np.nan
#             # do not update ground truth, let neighbor increase with next loop
#         else: 
#             ground_truth = neighbor

#     #new_nulls = paws_df_FILTERED['htu_temp'].isnull() & ~existing_nulls

#     paws_outliers = pd.concat([paws_outliers, pd.Series(outliers_to_add)], ignore_index=True)

# # -------------------------------------------------------------------------------------------------------------------------------
#     plt.figure(figsize=(20, 12))

#     plt.plot(paws_df_FILTERED['date'], paws_df_FILTERED['htu_temp'], marker='.', markersize=1, label=f"3D PAWS htu_temp")

#     plt.title(f'{station_directories[i][8:14]} Temperature: 3D PAWS htu_temp [AFTER CLEANING]')
#     plt.xlabel('Date')
#     plt.ylabel('Temperature (˚C)')
#     plt.xticks(rotation=45)

#     plt.legend()

#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(data_destination+station_directories[i]+"\\temperature\\TSMS00_htu-temp_AFTER.png")

#     plt.clf()
#     plt.close()
# # -------------------------------------------------------------------------------------------------------------------------------
