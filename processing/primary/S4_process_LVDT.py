"""
:module: S4_process_LVDT.py
:purpose: Conduct empirical melt corrections for LVDT measurements
		  to estimate relative ice-bed separation throughout
		  experiments T24, T6, and T96.

"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from glob import glob
sys.path.append(os.path.join('..','..'))
from util.datetimeindex import *

# Map directories
ROOT = os.path.join('..','..','..')
DDIR = os.path.join(ROOT,'processed','timeseries')
# Map data files
LVDT06H = os.path.join(DDIR,'S3_T06_LVDT_10min_x3_smoothed.csv')
LVDT24H = os.path.join(DDIR,'S3_T24_LVDT_10min_x3_smoothed.csv')
LVDT96H = os.path.join(DDIR,'S3_T96_LVDT_10min_x3_smoothed.csv')


### LOAD DATA ###
df_L06 = pd.read_csv(LVDT06H,parse_dates=True,index_col=[0])
df_L24 = pd.read_csv(LVDT24H,parse_dates=True,index_col=[0])
df_L96 = pd.read_csv(LVDT96H,parse_dates=True,index_col=[0])

isplot = False

### PICK  EXTREMUM ###
# Create mask to remove edges included in earlier processing step
DT_MASK = pd.Timedelta(6,unit='hour')
INDSS = df_L24.index <= df_L24.index.min() + DT_MASK
IND06 = (df_L06.index >= df_L06.index.min() + DT_MASK) & (df_L06.index <= df_L06.index.max() - DT_MASK)
IND24 = (df_L24.index >= df_L24.index.min() + DT_MASK) & (df_L24.index <= df_L24.index.max() - DT_MASK)
IND96 = (df_L96.index >= df_L96.index.min() + DT_MASK) & (df_L96.index <= df_L96.index.max() - DT_MASK)
T06_peaks = pick_extrema_indices(df_L06[IND06],T=pd.Timedelta(5,unit='hour'))
T24_peaks = pick_extrema_indices(df_L24[IND24],T=pd.Timedelta(23,unit='hour'))
T96_peaks = pick_extrema_indices(df_L96[IND96],T=pd.Timedelta(90,unit='hour'))

### CALCULATE STEADY STATE MELT RATE ###
mSS = fit_dateline(df_L24[INDSS].index,df_L24[INDSS].values)
print('The melt rate implied by steady-state is\n %.3e mm sec-1 (%.3e mm d-1)'%(mSS,mSS * 3600*24))


### CALCULATE MELT CORRECTIONS ###
m06_M = fit_dateline(T06_peaks['I_max'][-3:],T06_peaks['V_max'][-3:])
m06_m = fit_dateline(T06_peaks['I_min'][-3:],T06_peaks['V_min'][-3:])
m06_u = np.mean([m06_m,m06_M])
print('The melt rate being removed from T6 based on QSS is\n %.3e mm sec-1 (%.3e mm d-1)'%(m06_u,m06_u * 3600*24))
S_L06r = reduce_series_by_slope(df_L06.copy(),m06_u,T06_peaks['I_min'][-3],T06_peaks['V_min'][-3])


m24_M = fit_dateline(T24_peaks['I_max'][-2:],T24_peaks['V_max'][-2:])
m24_m = fit_dateline(T24_peaks['I_min'][-2:],T24_peaks['V_min'][-2:])
m24_u = np.mean([m24_m,m24_M])
print('The melt rate being removed from T24 based on QSS is\n %.3e mm sec-1 (%.3e mm d-1)'%(m24_u,m24_u * 3600*24))

S_L24r = reduce_series_by_slope(df_L24.copy(),m24_u,T24_peaks['I_min'][-2],T24_peaks['V_min'][-2])


S_L96r = df_L96 - np.nanmin(df_L96.values)
S_L96r = S_L96r.rename(columns={'LVDT_mm':'LVDT_mm red'})
print('Conducting only bulk-shift to T96 LVDT data: %.3e mm'%(np.nanmin(df_L96.values)))


## WRITE REDUCED SERIES TO FILE

S_L24r.to_csv(os.path.join(DDIR,'S4_T24_LVDT_10min_x3_smoothed_QSS_reduced.csv'),header=True,index=True)
S_L06r.to_csv(os.path.join(DDIR,'S4_T06_LVDT_10min_x3_smoothed_QSS_reduced.csv'),header=True,index=True)
S_L96r.to_csv(os.path.join(DDIR,'S4_T96_LVDT_10min_x3_smoothed_MIN_reduced.csv'),header=True,index=True)


if isplot:
	## Display results 
	fig = plt.figure()
	gs = GridSpec(ncols=3,nrows=2)
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[1,0])
	ax3 = fig.add_subplot(gs[0,1])
	ax4 = fig.add_subplot(gs[1,1])
	ax5 = fig.add_subplot(gs[0,2])
	ax6 = fig.add_subplot(gs[1,2])
	ax1.plot(df_L24)
	ax1.plot(T24_peaks['I_min'],T24_peaks['V_min'],'bo')
	ax1.plot(T24_peaks['I_max'],T24_peaks['V_max'],'ro')
	ax2.plot(S_L24r)

	ax3.plot(df_L06)
	ax3.plot(T06_peaks['I_min'],T06_peaks['V_min'],'bo')
	ax3.plot(T06_peaks['I_max'],T06_peaks['V_max'],'ro')
	ax4.plot(S_L06r)

	ax5.plot(df_L96)
	ax5.plot(T96_peaks['I_min'],T96_peaks['V_min'],'bo')
	ax5.plot(T96_peaks['I_max'],T96_peaks['V_max'],'ro')
	# ax6.plot(S_L96r)



	plt.show()
