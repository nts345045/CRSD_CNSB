import pandas as pd
import sys
import os
from tqdm import tqdm
"""
This module shifts all data to local Wisconsin time for October/November
and saves the CSV data with strings formatted for quicker read/write using
the Pandas library.

"""


# Define paths (OS-agnostic)
#### FILE PATHS & DATA LOADING WRAPPERS ####

ROOT = os.path.join('..','..','..')
# Define data sources
D_LVDT_UTC = os.path.join(ROOT,'processed','timeseries','UTC_LVDT_data_PD_thresh0.05mm.csv')
D_NTP_UTC = os.path.join(ROOT,'raw','NTauP','UTC_N_T_P.csv')
# Define output directory
ODIR = os.path.join(ROOT,'processed','timeseries')

DTL = pd.Timedelta(-5,unit='hour')

### LOAD DATA ###
df_NTP = pd.read_csv(D_NTP_UTC)
IDX = []
# Conduct timestamp reprocessing
for i_ in tqdm(range(len(df_NTP))):
	IDX.append(pd.Timestamp(df_NTP['TimeUTC'].values[i_]) + DTL)
df_NTP.index = IDX
# Subset data to relevant fields
df_NTP = df_NTP[['N_kPa','T_kPa','Pw1_kPa','Pw2_kPa']]
print('Stress data loaded')
print('Writing in Pandas Friendly Format (drastically accelerates I/O with pd.read_csv)')
# Save NTP dataframe to CSV
df_NTP.to_csv(os.path.join(ODIR,'S2_NTP_Local_PD_DateTime.csv'),header=True,index=True)

# Do quick update to LVDT and save to CSV
df_LVDT = pd.read_csv(D_LVDT_UTC,parse_dates=True,index_col=[0])
df_LVDT_out = df_LVDT['LVDT_mm_stitched']
df_LVDT_out.index += DTL
df_LVDT_out.to_csv(os.path.join(ODIR,'S2_LVDT_Local_PD_DateTime_FULL.csv'),header=True,index=True)



