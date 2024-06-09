"""
:module: S0_merge_LVDT.py
:purpose: merge individual LVDT raw data files into one CSV, filtering
for large jumps and correcting these jumps.

:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
"""

from glob import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


ROOT = os.path.join('..','..','..')
IROOT = os.path.join(ROOT,'raw','LVDT')
OROOT = os.path.join(ROOT,'processed','timeseries')

### PULL RAW LVDT FILES ###
flist = glob(os.path.join(IROOT,'*.txt'))
### UPDATE TIMES TO PANDAS FRIENDLY DATETIMES ###
T0 = pd.Timestamp("1904-01-01")
df_ = pd.DataFrame()
for f_ in tqdm(flist):
	_df_ = pd.read_csv(f_,delim_whitespace=True)
	IDX = []
	for i_ in _df_['time']:
		IDX.append(pd.Timedelta(i_,unit='sec') + T0)
	_df_.index = IDX
	df_ = pd.concat([df_,_df_],axis=0,ignore_index=False)

### CONDUCT FIRST-DIFFERENCE FILTERING & STITCHING ###
thresh = 0.05
rng = 1
# Sort indices
df_ = df_.sort_index()
# Grab flipped LVDT
S_LVDT = df_['LVDT1.1']
# Grab data
zv = S_LVDT.copy().values
# Create finite differences to detect large jumps
dz = zv[rng:] - zv[:-rng]
# Apply progressive shifts to trailing LVDT measurements
for i_,dz_ in enumerate(dz):
	if np.abs(dz_) > thresh:
		zv[i_:] -= dz_

# merge back into dataframe & write to disk
df_ = pd.concat([df_,pd.Series(zv,index=df_.index,name='LVDT_mm_stitched')],axis=1,ignore_index=False)
OFILE = os.path.join(OROOT,'UTC_LVDT_data_PD_thresh%0.2fmm.csv'%(thresh))
df_.to_csv(OFILE,index=True,header=True)
