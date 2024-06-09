"""
:module: JGLAC_Fig05_Experiment_T24_Timeseries.py
:purpose: Plot the key timeseries from experiment T24 (24 hour oscillation)
:version: 0 - Submission format to Journal of Glaciology
:short ref: Stevens, Hansen, and others
			Experimental constraints on transient glacier sliding with ice-bed separation
:Figure #: 5
:Figure Caption: Experiment T24 temporal evolution of 
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
T24_NT = os.path.join(DDIR,'S3_T24_NT_10min_x3_smoothed.csv')
T24_LV = os.path.join(DDIR,'S4_T24_LVDT_10min_x3_smoothed_QSS_reduced.csv')
T24_CM = os.path.join(DDIR,'S5_experiment_T24_cavity_metrics.csv')
T24_CP = os.path.join(ROOT,'processed','Cavity_Picks','Postprocessed_Contact_Geometries.csv')
# Map output directory
ODIR = os.path.join(ROOT,'results','figures','manuscript')

# LOAD EXPERIMENTAL DATA #
df_NT24 = pd.read_csv(T24_NT,parse_dates=True,index_col=[0])
df_Z24 = pd.read_csv(T24_LV,parse_dates=True,index_col=[0])
df_CM24 = pd.read_csv(T24_CM,parse_dates=True,index_col=[0])
df_CP24 = pd.read_csv(T24_CP,parse_dates=True,index_col=[0])

# Plotting Controls #
issave = True
DPI = 200
FMT = 'PNG'

# Define Reference time for T24
t0_T24 = pd.Timestamp('2021-10-26T13:58')
D_tau = 0 #60.95 # [kPa] difference between observed \tau during steady state
			  #       minus calculated values given the geometry of the bed
			  # 	  for steady-state N(t)

D_tauP = df_CM24[df_CM24.index > t0_T24 + pd.Timedelta(121,unit='hour')]['T kPa'].mean() - \
		df_CM24[df_CM24.index > t0_T24 + pd.Timedelta(121,unit='hour')]['hat T kPa'].mean()
print('Estimated shift for \\tau is %.3e kPa'%(D_tauP))
# Make elapsed time indices
# For stress measures
dtindex = (df_NT24.index - t0_T24).total_seconds()/3600
# For LVDT measures
dtzindex = (df_Z24.index - t0_T24).total_seconds()/3600


### CALCULATE OBSERVED DRAG ###
# Observed Drag with \Delta \tau offset
mu_obs = (df_NT24['T_kPa'].values - D_tauP)/df_NT24['N_kPa'].values
df_mu_obs = pd.DataFrame({'mu_obs':mu_obs},index=df_NT24.index)
# Get Reference \\mu(t) value for differencing
mu0_obs = df_mu_obs[dtindex >= 121 ].mean().values
# Get observed change in drag
Dmu_obs = mu_obs - mu0_obs

### CALCULATE SHIFTED DRAG ###
mu_tP = (df_NT24['T_kPa'].values - D_tauP)/df_NT24['N_kPa'].values


### CALCULATE STEADY-STATE MODEL DRAG ###
mu_calc = df_CM24['hat T kPa'].values/df_CM24['N kPa'].values
# Get reference \\mu(t) for differencing
mu0_calc = mu_calc[-1]
# Get calculated change in drag
Dmu_calc = mu_calc - mu0_calc


### PLOTTING SECTION ###
# Initialize figure and subplot axes
fig,axs = plt.subplots(ncols=1,nrows=3,figsize=(7.5,5.5))


# (a) PLOT \tau(t) 
# Plot observed values
axs[0].plot(dtindex,df_NT24['T_kPa'] - D_tau,'k-',zorder=10)
# Plot modeled values
axs[0].plot(dtzindex,df_CM24['hat T kPa'].values,'r--',zorder=5)
axs[0].plot(dtindex,df_NT24['T_kPa'] - D_tauP,'b-',zorder=8)
# Apply labels & formatting
axs[0].set_ylabel('Shear Stress\n[$\\tau$] (kPa)')
# axb.set_ylabel('$\\hat{\\tau}$(t) [kPa]',rotation=270,labelpad=15,color='red')
axs[0].set_xticks(np.arange(0,132,12))
axs[0].grid(axis='x',linestyle=':')
axs[0].text(119,df_CM24['hat T kPa'].values[-1] + D_tauP/2,'$\\Delta \\tau$',fontsize=14,ha='center',va='center')
axs[0].arrow(123,163,0,95.4 - 164,head_width=2,width=0.1,head_length=10,fc='k')


# (b) PLOT \Delta \mu(t)
# Plot observed values
# axs[1].plot(dtindex,mu_obs,'k-',zorder=10,label='Obs.')
# plot modeled values
axs[1].plot(dtzindex,mu_calc,'r--',zorder=5,label='Mod.')
axs[1].plot(dtindex,mu_tP,'b-',zorder=5,label='$\\tau_{adj}$')
# Apply labels & formatting
axs[1].set_xticks(np.arange(0,132,12))
axs[1].grid(axis='x',linestyle=':')
axs[1].set_ylabel('Drag\n[$\\mu$] ( - )')
# axc.set_ylabel('$\\Delta\\mu$ (t) [ - ]',rotation=270,labelpad=15,color='red')
# axs[1].set_ylim([-0.11,0.22])
# axs[1].legend(ncol=2,loc='upper left')

# (c) PLOT S(t)
# Plot mapped values from LVDT
axs[2].plot(dtzindex,df_CM24['S tot'],'b-',zorder=10)
# Plot modeled values
axs[2].plot(dtzindex,df_CM24['hat S tot'],'r--',zorder=5)
# Apply labels and formatting
axs[2].set_xticks(np.arange(0,132,12))
axs[2].grid(axis='x',linestyle=':')
axs[2].set_ylabel('Scaled Contact\nLength [$S$] ( - )')

# # (d) PLOT R(t)
# # Plot observed values
# axs[3].plot(dtzindex,df_CM24['R mea'].values,'k-',zorder=10)
# axs[3].set_xticks(np.arange(0,132,12))
# axs[3].set_ylabel('$R(t)$ [ - ]')
# axs[3].plot(dtzindex,df_CM24['hat R mea'],'r--',zorder=5)

# axs[3].set_xlabel('Experiment runtime [h]')
# axs[3].grid(axis='x',linestyle=':')
# ## SUBPLOT FORMATTING
plt.subplots_adjust(hspace=0)
LBL = ['a','b','c','d']
DAT = (df_NT24['T_kPa'],df_mu_obs['mu_obs'],\
	   -df_Z24['LVDT_mm_stitched red'])

for i_,D_ in enumerate(DAT):
	# Pick extremum within given period
	ex24 = pick_extrema_indices(D_,T=pd.Timedelta(24,unit='hour'))
	# Get y-lims
	ylim = axs[i_].get_ylim()
	# Plot minima times
	for t_ in ex24['I_min']:
		axs[i_].plot(np.ones(2,)*(t_ - t0_T24).total_seconds()/3600,ylim,\
					 ':',color='dodgerblue',zorder=1)
	# Plot maxima times
	for t_ in ex24['I_max']:
		axs[i_].plot(np.ones(2,)*(t_ - t0_T24).total_seconds()/3600,ylim,\
					 '-.',color='dodgerblue',zorder=1)
	# re-enforce initial ylims
	axs[i_].set_ylim(ylim)
	# set xlims from data limits
	axs[i_].set_xlim((dtindex.min(),dtindex.max()))
	axs[i_].text(-1,(ylim[1] - ylim[0])*0.85 + ylim[0],\
				 LBL[i_],fontsize=14,fontweight='extra bold',\
				 fontstyle='italic',ha='right')

axs[-1].set_xlabel('Elapsed Time (hr)')

if issave:
	OFILE = os.path.join(ODIR,'JGLAC_Fig05_v1.6_Experiment_T24_Timeseries_%ddpi.%s'%(DPI,FMT.lower()))
	plt.savefig(OFILE,dpi=DPI,format=FMT.lower())

plt.show()

