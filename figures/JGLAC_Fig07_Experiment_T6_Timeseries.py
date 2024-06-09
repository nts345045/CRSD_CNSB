"""
:module: JGLAC_Fig05_Experiment_T06_Timeseries.py
:purpose: Plot the key timeseries from experiment T06 (06 hour oscillation)
:version: 0 - Submission format to Journal of Glaciology
:short ref: Stevens, Hansen, and others
			Experimental constraints on transient glacier sliding with ice-bed separation
:Figure #: 5
:Figure Caption: Experiment T06 temporal evolution of 
				 (a) effective pressure, N(t), 
				 (b) shear stress, \tau\left(t\right), 
				 (c) relative ice-bed separation, \Delta z^\ast\left(t\right), and 
				 (d) relative drag, \Delta\mu\left(t\right). 
				 Vertical bars mark the timing of maximum (solid) and minimum (dashed) values of 
				 	N(t) (blue),
				  	\tau\left(t\right) (orange), 
				  	\Delta z^\ast\left(t\right) (red), and 
				  	\Delta\mu\left(t\right) (violet) within each cycle. 

:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join('..'))
from util.datetimeindex import pick_extrema_indices
import core.model.steadystate_model as ssm

# Map Directories
ROOT = os.path.join('..','..')
DDIR = os.path.join(ROOT,'processed','timeseries')
# Map Experimental Data Files
T06_NT = os.path.join(DDIR,'S3_T06_NT_10min_x3_smoothed.csv')
T06_LV = os.path.join(DDIR,'S4_T06_LVDT_10min_x3_smoothed_QSS_reduced.csv')
T06_CM = os.path.join(DDIR,'S5_experiment_T6_cavity_metrics.csv')
T06_CP = os.path.join(ROOT,'processed','Cavity_Picks','Postprocessed_Contact_Geometries.csv')
# Map output directory
ODIR = os.path.join(ROOT,'results','figures','manuscript')

# LOAD EXPERIMENTAL DATA #
df_NT06 = pd.read_csv(T06_NT,parse_dates=True,index_col=[0])
df_Z06 = pd.read_csv(T06_LV,parse_dates=True,index_col=[0])
df_CM06 = pd.read_csv(T06_CM,parse_dates=True,index_col=[0])
df_CP06 = pd.read_csv(T06_CP,parse_dates=True,index_col=[0])

# Plotting Controls #
issave = True
DPI = 200
FMT = 'PNG'

# Define Reference time for T06
t0_T06 = pd.Timestamp('2021-11-1T11:09:15')
D_tau = 0 #60.95 # [kPa] difference between observed \tau during steady state
			  #       minus calculated values given the geometry of the bed
			  # 	  for steady-state N(t)

D_tauP = np.nanmean(df_CM06[df_CM06.index > t0_T06 + pd.Timedelta(30.5,unit='hour')]['T kPa']) - \
		 np.nanmean(df_CM06[df_CM06.index > t0_T06 + pd.Timedelta(30.5,unit='hour')]['hat T kPa'])
print('Estimated shift for \\tau is %.3e kPa'%(D_tauP))

# Make elapsed time indices
# For stress measures
dtindex = (df_NT06.index - t0_T06).total_seconds()/3600
# For LVDT measures
dtzindex = (df_Z06.index - t0_T06).total_seconds()/3600
dtmindex = (df_CM06.index - t0_T06).total_seconds()/3600

### CALCULATE OBSERVED DRAG ###
# Observed Drag with \Delta \tau offset
mu_obs = (df_NT06['T_kPa'].values - D_tauP)/df_NT06['N_kPa'].values
df_mu_obs = pd.DataFrame({'mu_obs':mu_obs},index=df_NT06.index)
# Get Reference \\mu(t) value for differencing
mu0_obs = df_mu_obs[dtindex <= 0].mean().values
# Get observed change in drag
Dmu_obs = mu_obs - mu0_obs

### CALCULATE STEADY-STATE MODEL DRAG ###
mu_calc = df_CM06['hat T kPa'].values/df_CM06['N kPa'].values
# Get reference \\mu(t) for differencing
mu0_calc = mu_calc[-1]
# Get calculated change in drag
Dmu_calc = mu_calc - mu0_calc

### CALCULATE SHIFTED DRAG ###
mu_tP = (df_NT06['T_kPa'].values - D_tauP)/df_NT06['N_kPa'].values



### PLOTTING SECTION ###
# Initialize figure and subplot axes
fig,axs = plt.subplots(ncols=1,nrows=3,figsize=(7.5,5.5))


# (a) PLOT \tau(t) 
# Plot observed values
axs[0].plot(dtindex,df_NT06['T_kPa'],'k-',zorder=10)
# Plot modeled values
axs[0].plot(dtmindex,df_CM06['hat T kPa'].values,'r--',zorder=5)  
axs[0].plot(dtindex,df_NT06['T_kPa'] - D_tauP,'b-',zorder=7)
# Apply labels & formatting
axs[0].set_ylabel('Shear Stress\n[$\\tau$] (kPa)')
# axb.set_ylabel('$\\hat{\\tau}$(t) [kPa]',rotation=270,labelpad=15,color='red')
axs[0].set_xticks(np.arange(-3,36,3))
axs[0].grid(axis='x',linestyle=':')

axs[0].text(-0.25,df_CM06['hat T kPa'].values[-1] + D_tauP/2,'$\\Delta \\tau$',fontsize=14,ha='center',va='center')
axs[0].arrow(-1.33,162.33,0,95.4 - 162.33,head_width=2/5,width=0.1/5,head_length=10,fc='k')


# (b) PLOT \Delta \mu(t)
# Plot observed values
# axs[1].plot(dtindex,mu_obs,'k-',zorder=10,label='Obs.')
# plot modeled values
axs[1].plot(dtmindex,mu_calc,'r--',zorder=5,label='Mod.')
# Shifted \tau result
axs[1].plot(dtindex,mu_tP,'b-',zorder=7,label='$\\tau_{adj}$')
# Apply labels & formatting
axs[1].set_xticks(np.arange(-3,36,3))
axs[1].grid(axis='x',linestyle=':')
axs[1].set_ylabel('Drag\n[$\\mu$] ( - )')
# axc.set_ylabel('$\\Delta\\mu$ (t) [ - ]',rotation=270,labelpad=15,color='red')
# axs[1].set_ylim([-0.11,0.22])
# axs[1].legend(ncol=1,loc='upper left')

# (c) PLOT S(t)
# Plot mapped values from LVDT
axs[2].plot(dtmindex,df_CM06['S tot'],'b-',zorder=10)
# Plot modeled values
axs[2].plot(dtmindex,df_CM06['hat S tot'],'r--',zorder=5)
# Apply labels and formatting
axs[2].set_xticks(np.arange(-3,36,3))
axs[2].grid(axis='x',linestyle=':')
axs[2].set_ylabel('Scaled Contact Length\n[$S$] ( - )')


# ## SUBPLOT FORMATTING
plt.subplots_adjust(hspace=0)
LBL = ['a','b','c']
DAT = (df_NT06['T_kPa'],df_mu_obs['mu_obs'],\
	   -df_Z06['LVDT_mm_stitched red'])

ex_lims = [t0_T06 + pd.Timedelta(15,unit='minute'), t0_T06 + pd.Timedelta(28,unit='hour')]

for i_,D_ in enumerate(DAT):
	axs[i_].set_xlim([-2.9,35])
	# Pick extremum within given period
	ex6 = pick_extrema_indices(D_,T=pd.Timedelta(5.5,unit='hour'))
	# Get y-lims
	ylim = axs[i_].get_ylim()
	# Plot minima times
	for t_ in ex6['I_min']:
		if ex_lims[0] <= t_ <= ex_lims[1]:
			axs[i_].plot(np.ones(2,)*(t_ - t0_T06).total_seconds()/3600,ylim,\
						 ':',color='dodgerblue',zorder=1)
	# Plot maxima times
	for t_ in ex6['I_max']:
		if ex_lims[0] <= t_ <= ex_lims[1]:
			axs[i_].plot(np.ones(2,)*(t_ - t0_T06).total_seconds()/3600,ylim,\
						 '-.',color='dodgerblue',zorder=1)
	# re-enforce initial ylims
	axs[i_].set_ylim(ylim)
	# set xlims from data limits
	# axs[i_].set_xlim((dtindex.min(),dtindex.max()))
	axs[i_].text(33.5,(ylim[1] - ylim[0])*0.875 + ylim[0],\
				 LBL[i_],fontsize=14,fontweight='extra bold',\
				 fontstyle='italic')
	if LBL[i_] != 'c':
		axs[i_].xaxis.set_tick_params(labelbottom=False)



axs[-1].set_xlabel('Elapsed Time (hr)')

if issave:
	OFILE = os.path.join(ODIR,'JGLAC_Fig07_v1.6_Experiment_T06_Timeseries_%ddpi.%s'%(DPI,FMT.lower()))
	plt.savefig(OFILE,dpi=DPI,format=FMT.lower())

plt.show()

