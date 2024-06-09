"""
:module: JGLAC_Fig3_T24_T6_Nt_Plot.py
:purpose: Plot the N(t) forcing function starting from the N(t) steady
		  state prior to experiment T24 through to the end of the 
		  steady-state following experiment T6
:version: 0 - Submission format to Journal of Glaciology
:short ref: Stevens, Hansen, and others
			Experimental constraints on transient glacier sliding with ice-bed separation
:Figure #: 3
:Figure Caption: Effective pressure (vertical pressure – water pressure) profiles with experiment 
				 names (below) and cycle numbers (above) labeled. The steady state periods and 
				 experimental periods are delimited with vertical dashed lines. Timestamps are in 
				 central US standard time (UTC – 5 hours).
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu

"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from glob import glob

issave = True
DPI = 200
FMT = 'PNG'

ROOT = os.path.join('..','..')
DDIR = os.path.join(ROOT,'processed','timeseries')
D_NT = os.path.join(DDIR,'S3_FULL_NT_10min_x3_smoothed.csv')
ODIR = os.path.join(ROOT,'results','figures','manuscript')

df = pd.read_csv(D_NT,parse_dates=True,index_col=[0])


LBV = ['Mechanically Inferred\nSteady State','T24','Geometric Steady State\n(Rest Period)','T06']#,'T96']
TSV = [pd.Timestamp("2021-10-25T13:00:00"),pd.Timestamp('2021-10-26T13:56'),\
	   pd.Timestamp("2021-10-31T14:03"),pd.Timestamp('2021-11-1T11:09:15')]#,pd.Timestamp("2021-11-03T15:52")]
TEV = [pd.Timestamp("2021-10-26T13:56"),pd.Timestamp("2021-10-31T14:03"),\
	   pd.Timestamp('2021-11-1T11:09:15'),pd.Timestamp("2021-11-02T17:26:30")]
#	   pd.Timestamp("2021-11-02T17:26:30"),pd.Timestamp("2021-11-22T12:09")]

xlims = [pd.Timestamp('2021-10-25'),TEV[-1] + pd.Timedelta(3,unit='hour')]



fig = plt.figure(figsize=(7.5,5.3))
ax1 = fig.add_subplot(111)

ax1.plot(df['N_kPa'],'k-')

ylims = ax1.get_ylim()

# Insert experiment labels
for i_,l_ in enumerate(LBV):
	ax1.plot([TSV[i_],TSV[i_]],ylims,'k:',alpha=0.5)
	ax1.plot([TEV[i_],TEV[i_]],ylims,'k:',alpha=0.5)
	if i_ in [0,2]:
		ax1.text(TSV[i_] + (TEV[i_] - TSV[i_])/2,275,LBV[i_],ha='center',va='center',rotation=90)
	else:
		ax1.text(TSV[i_] + (TEV[i_] - TSV[i_])/2,180,'Exp. {}'.format(LBV[i_]),ha='center')

# Insert cycle numbering
# T96C = [pd.Timestamp("2021-11-04T12"),pd.Timestamp("2021-11-06T12"),pd.Timestamp("2021-11-10T12"),\
# 		pd.Timestamp("2021-11-14T20"),pd.Timestamp("2021-11-19T13")]

for i_ in range(5):
	ax1.text(TSV[1] + pd.Timedelta(6,unit='hour') + i_*pd.Timedelta(24,unit='hour'),510,i_+1,ha='center')
	ax1.text(TSV[3] + pd.Timedelta(3,unit='hour') + i_*pd.Timedelta(6,unit='hour'),510,i_ + 1, ha='center')


ax1.set_xlim(xlims)
ax1.set_ylim([170,530])
ax1.set_ylabel('Effective Pressure [$N$] (kPa)')
ax1.set_xlabel('October 2021                            Day of Month'+\
			   '                            November 2021\nLocal Time (UTC - 5)')

ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=np.arange(0,25,3,dtype=int)))
ax1.format_xdata = mdates.DateFormatter('%d')


if issave:
	OFILE = os.path.join(ODIR,'JGLAC_Fig03_v1.6_Effective_Pressure_Labeled_%ddpi.%s'%(DPI,FMT.lower()))
	plt.savefig(OFILE,dpi=DPI,format=FMT.lower())

plt.show()