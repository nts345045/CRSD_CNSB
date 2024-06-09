"""
:module: JGLAC_Fig07_CrossPlots_MAIN.py
:purpose: Plot the key timeseries from experiment T6 (6 hour oscillation)
:version: 0 - Submission format to Journal of Glaciology
:short ref: Stevens, Hansen, and others
			Experimental constraints on transient glacier sliding with ice-bed separation
:Figure #: 7
:Figure Caption: Parameter cross-plots for experiment T24 & T6
					(a–b) \\tau(N(t))
					(c–d) \\Delta \\mu(N(t))
					(e–f) \\tau(S(t))
					(g-h) \\Delta \\mu(S(t)) 
				Cycle number and progress through each cycle is denoted with line color 
				(see color bar). Comparable modeled values from the double-valued sliding 
				rule of Zoet and Iverson (2015) are shown in red in (a), (c), and (e).
			  	 
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Plotting Routine ##
def plot_cycles(axis,Xdata,Ydata,Tindex,t0,cmaps,ncycles=5,T=pd.Timedelta(24,unit='hour'),zorder=10):
	chs = []
	for I_ in range(ncycles):
		TS = t0 + I_*T
		TE = t0 + (I_ + 1)*T
		IND = (Tindex >= TS) & (Tindex < TE)
		XI = Xdata[IND]
		YI = Ydata[IND]
		cbl = axis.scatter(XI,YI,c=(Tindex[IND] - TS)/T,cmap=cmaps[I_],s=1,zorder=zorder)
		chs.append(cbl)
	return chs

# Map Data
ROOT = os.path.join('..','..')
DDIR = os.path.join(ROOT,'processed','timeseries')
# Map Experimental Data
T24_CM = os.path.join(DDIR,'S5_experiment_T24_cavity_metrics.csv')
T6_CM = os.path.join(DDIR,'S5_experiment_T6_cavity_metrics.csv')
MOD_CM = os.path.join(DDIR,'S5_modeled_values.csv')

# Map output directory
ODIR = os.path.join(ROOT,'results','figures','manuscript')

### LOAD EXPERIMENTAL DATA ###
# Load Experimental Data
df_T06 = pd.read_csv(T6_CM,parse_dates=True,index_col=[0])
df_T24 = pd.read_csv(T24_CM,parse_dates=True,index_col=[0])
df_MOD = pd.read_csv(MOD_CM,parse_dates=True,index_col=[0])

### SET REFERENCES AND OFFSETS ###
t0_T24 = pd.Timestamp('2021-10-26T13:58')
t0_T06 = pd.Timestamp('2021-11-1T11:09:15')
D_tau = 0 #60.95 # [kPa] - amount to reduce \tau(t)

### SET PLOTTING PARAMETERS
cmaps = ['Blues_r','Purples_r','RdPu_r','Oranges_r','Greys_r']
issave = True
DPI = 200
FMT = 'PNG'
URC = (0.9,0.85)
ULC = (0.05,0.90)
PADXY = 0.05

### PLOTTING SECTION ###
fig = plt.figure(figsize=(7.5,10))
GS1 = fig.add_gridspec(ncols=2,nrows=43,hspace=0,wspace=0)
GS2 = fig.add_gridspec(ncols=2,nrows=43,hspace=0,wspace=0)
axs = [fig.add_subplot(GS1[0:11,0]),fig.add_subplot(GS1[0:11,1]),\
	   fig.add_subplot(GS1[11:21,0]),fig.add_subplot(GS1[11:21,1]),\
	   fig.add_subplot(GS2[24:34,0]),fig.add_subplot(GS2[24:34,1]),\
	   fig.add_subplot(GS2[34:,0]),fig.add_subplot(GS2[34:,1])]



### (a) Plot T24 \tau(N(t)) ###
XI = df_T24['N kPa'].values
YI = df_T24['T kPa'].values# - D_tau
II = df_T24.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))

# Observed
chs = plot_cycles(axs[0],XI,YI,II,\
				  t0_T24,cmaps,ncycles=5,\
				  T=pd.Timedelta(24,unit='hour'),zorder=10)
# Modeled
axs[0].plot(df_MOD.index.values*1e-3,df_MOD['T_Pa'].values*1e-3,\
			  'r--',zorder=5)

axs[0].set_xlim(xlims)
axs[0].set_ylim(ylims)
# Axis labels
axs[0].set_ylabel('$\\tau(N(t))$ [kPa]')
axs[0].set_xlabel('$N(t)$ [kPa]')
# Subplot Label
# axs[0].text(xlims[0] + ULC[0]*(xlims[1] - xlims[0]),\
# 			  ylims[0] + ULC[1]*(ylims[1] - ylims[0]),\
# 			  'a',fontsize=14,fontweight='extra bold',\
# 			  fontstyle='italic',ha='center',va='center')
axs[0].set_title('Experiment T24')



### (b) Plot T6 \tau(N(t)) ###
XI = df_T06['N kPa'].values
YI = df_T06['T kPa'].values#  - D_tau
II = df_T06.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))

# Observed
plot_cycles(axs[1],XI,YI,II,\
			t0_T06,cmaps,ncycles=5,\
			T=pd.Timedelta(6,unit='hour'),zorder=10)
# Modeled
axs[1].plot(df_MOD.index.values*1e-3,df_MOD['T_Pa'].values*1e-3,\
			  'r--',zorder=5)

axs[1].set_xlim(xlims)
axs[1].set_ylim(ylims)
# Axis labels
axs[1].set_ylabel('$\\tau(N(t))$ [kPa]',rotation=270,labelpad=15)
axs[1].set_xlabel('$N(t)$ [kPa]')
# Subplot Label
# axs[1].text(xlims[0] + ULC[0]*(xlims[1] - xlims[0]),\
# 			  ylims[0] + ULC[1]*(ylims[1] - ylims[0]),\
# 			  'b',fontsize=14,fontweight='extra bold',\
# 			  fontstyle='italic',ha='center',va='center')
axs[1].set_title('Experiment T06')




### (c) Plot T24 \Delta \mu(N(t)) ###
XI = df_T24['N kPa'].values
mu_obs = (df_T24['T kPa'].values)/XI
mu0_obs = np.mean(mu_obs[df_T24.index < t0_T24])
YI = mu_obs - mu0_obs
II = df_T24.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))
# Observed
plot_cycles(axs[2],XI,YI,II,\
			t0_T24,cmaps,ncycles=5,\
			T=pd.Timedelta(24,unit='hour'),zorder=10)


# Modeled
mu_calc = df_MOD['T_Pa'].values/df_MOD.index.values
mu0_calc = mu_calc[0]
Dmu_calc = mu_calc - mu0_calc
# mu0_calc = df_T24['hat T kPa'].values[0]/df_T24['N kPa'].values[0]
# mu_calc = (df_T24.sort_values('N kPa')['hat T kPa'].values)/\
# 		   df_T24.sort_values('N kPa')['N kPa'].values
# Dmu_calc = mu_calc# - mu0_calc
axs[2].plot(df_MOD.index.values*1e-3,Dmu_calc,\
			  'r--',zorder=5)
axs[2].set_ylabel('$\\Delta\\mu(N(t))$ [ - ]')
axs[2].set_xlabel('$N(t)$ [kPa]')
axs[2].set_xlim(xlims)
axs[2].set_ylim(ylims)

# axs[2].text(xlims[0] + URC[0]*(xlims[1] - xlims[0]),\
# 			  ylims[0] + URC[1]*(ylims[1] - ylims[0]),\
# 			  'c',fontsize=14,fontweight='extra bold',\
# 			  fontstyle='italic')




### (d) Plot T6 \Delta \mu(N(t)) ###
XI = df_T06['N kPa'].values
mu_obs = (df_T06['T kPa'].values)/XI
mu0_obs = np.mean(mu_obs[df_T06.index < t0_T06])
YI = mu_obs - mu0_obs
II = df_T06.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))
# Observed
plot_cycles(axs[3],XI,YI,II,\
			t0_T06,cmaps,ncycles=5,\
			T=pd.Timedelta(6,unit='hour'),zorder=10)


# Modeled
mu_calc = df_MOD['T_Pa'].values/df_MOD.index.values
Dmu_calc = mu_calc - mu_calc[0]
axs[3].plot(df_MOD.index.values*1e-3,Dmu_calc,\
			  'r--',zorder=5)
axs[3].set_ylabel('$\\Delta\\mu(N(t))$ [ - ]',rotation=270,labelpad=15)
axs[3].set_xlabel('$N(t)$ [kPa]')
axs[3].set_xlim(xlims)
axs[3].set_ylim(ylims)

# axs[3].text(xlims[0] + URC[0]*(xlims[1] - xlims[0]),\
# 			  ylims[0] + URC[1]*(ylims[1] - ylims[0]),\
# 			  'd',fontsize=14,fontweight='extra bold',\
# 			  fontstyle='italic')





### (e) Plot T24 \tau(S(t)) ###
XI = df_T24['S tot'].values
YI = df_T24['T kPa'].values# - D_tau
II = df_T24.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))

plot_cycles(axs[4],XI,YI,II,\
			t0_T24,cmaps,ncycles=5,\
			T=pd.Timedelta(24,unit='hour'),zorder=10)
# Plot Modeled Values
axs[4].plot(df_MOD['Stot'].values,df_MOD['T_Pa'].values*1e-3,\
			  'r--',zorder=5)

axs[4].set_ylabel('$\\tau(S(t))$ [kPa]')
axs[4].set_xlabel('$S(t)$ [ - ]')
axs[4].set_xlim(xlims)
axs[4].set_ylim(ylims)

# axs[4].text(xlims[0] + URC[0]*(xlims[1] - xlims[0]),\
# 			  ylims[0] + URC[1]*(ylims[1] - ylims[0]),\
# 			  'e',fontsize=14,fontweight='extra bold',\
# 			  fontstyle='italic')


### (f) Plot T6 \tau(S(t)) ###
XI = df_T06['S tot'].values
YI = df_T06['T kPa'].values# - D_tau
II = df_T06.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))
# Observed
plot_cycles(axs[5],XI,YI,II,\
			t0_T06,cmaps,ncycles=5,\
			T=pd.Timedelta(6,unit='hour'),zorder=10)
# Modeled values
axs[5].plot(df_MOD['Stot'].values,df_MOD['T_Pa'].values*1e-3,\
			  'r--',zorder=5)

axs[5].set_ylabel('$R(\\tau(t))$ [ - ]',rotation=270,labelpad=15)
axs[5].set_xlabel('$\\tau(t)$ [kPa]')
axs[5].set_xlim(xlims)
axs[5].set_ylim(ylims)
# axs[5].text(xlims[0] + URC[0]*(xlims[1] - xlims[0]),\
# 			  ylims[0] + URC[1]*(ylims[1] - ylims[0]),\
# 			  'f',fontsize=14,fontweight='extra bold',\
# 			  fontstyle='italic')


### (g) Plot T24 \Delta\mu(S(t)) ###
XI = df_T24['S tot'].values# - D_tau
mu_obs = df_T24['T kPa'].values/df_T24['N kPa'].values
mu0_obs = np.mean(mu_obs[df_T24.index < t0_T24])
YI = mu_obs - mu0_obs
# YI = df_T24['R mea'].values
II = df_T24.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))

plot_cycles(axs[6],XI,YI,II,\
			t0_T24,cmaps,ncycles=5,\
			T=pd.Timedelta(24,unit='hour'),zorder=10)
# Plot Modeled Values
mu_calc = df_MOD['T_Pa'].values/df_MOD.index.values
Dmu_calc = mu_calc - mu_calc[0]
axs[6].plot(df_MOD['Stot'].values,Dmu_calc,\
			  'r--',zorder=5)

axs[6].set_ylabel('$\\Delta\\mu(S(t))$ [ - ]')
axs[6].set_xlabel('$S(t)$ [ - ]')
axs[6].set_xlim(xlims)
axs[6].set_ylim(ylims)

# axs[6].text(xlims[0] + URC[0]*(xlims[1] - xlims[0]),\
# 			  ylims[0] + URC[1]*(ylims[1] - ylims[0]),\
# 			  'g',fontsize=14,fontweight='extra bold',\
# 			  fontstyle='italic')


### (h) Plot T6 R(\tau(t)) ###
XI = df_T06['S tot'].values# - D_tau
mu_obs = df_T06['T kPa'].values/df_T06['N kPa'].values
mu0_obs = np.mean(mu_obs[df_T06.index < t0_T06])
YI = mu_obs - mu0_obs
II = df_T06.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))
# Observed
plot_cycles(axs[7],XI,YI,II,\
			t0_T06,cmaps,ncycles=5,\
			T=pd.Timedelta(6,unit='hour'),zorder=10)
# Modeled values
mu_calc = df_MOD['T_Pa'].values/df_MOD.index.values
Dmu_calc = mu_calc - mu_calc[0]
axs[7].plot(df_MOD['Stot'].values,Dmu_calc,\
			  'r--',zorder=5)

axs[7].set_ylabel('$\\Delta\\mu(S(t))$ [ - ]')#,rotation=270,labelpad=15)
axs[7].set_xlabel('$S(t)$ [ - ]')
axs[7].set_xlim(xlims)
axs[7].set_ylim(ylims)
# axs[7].text(xlims[0] + URC[0]*(xlims[1] - xlims[0]),\
# 			  ylims[0] + URC[1]*(ylims[1] - ylims[0]),\
# 			  'h',fontsize=14,fontweight='extra bold',\
# 			  fontstyle='italic')

lblkw = {'fontsize':14,'fontweight':'extra bold','fontstyle':'italic',\
		 'ha':'center','va':'center'}

for i_,lbl_ in enumerate(['a','b','c','d','e','f','g','h']):
	xlims = axs[i_].get_xlim()
	ylims = axs[i_].get_ylim()

	if i_%2 == 1:
		axs[i_].yaxis.set_visible(False)
		xlims = axs[i_ - 1].get_xlim()
		ylims = axs[i_ - 1].get_ylim()
		axs[i_].set_xlim(xlims)
		axs[i_].set_ylim(ylims)

	if i_ in [0,1,4,5]:
		axs[i_].xaxis.set_visible(False)
		axs[i_].text(xlims[0] + (xlims[1] - xlims[0])*0.05,\
					 ylims[0] + (ylims[1] - ylims[0])*0.9,\
					 lbl_,**lblkw)
	if i_ in [2,3,6,7]:
		axs[i_].text(xlims[0] + (xlims[1] - xlims[0])*0.95,\
					 ylims[0] + (ylims[1] - ylims[0])*0.9,\
					 lbl_,**lblkw)




### FORMATTING SECTION ###
# plt.subplots_adjust(wspace=0)
# for I_ in range(axs.shape[0]):
# 	for J_ in range(axs.shape[1]):
# 		if J_ == 1:
# 			axs[I_,J_].yaxis.set_label_position('right')
# 			axs[I_,J_].yaxis.set_ticks_position('right')

### COLORBAR PLOTTING ###

Tc = 6

# Create timing colorbar
for k_ in range(5):
	cax = fig.add_axes([.15 + (.70/5)*k_,.045,.70/5,.015])
	chb = plt.colorbar(chs[k_],cax=cax,orientation='horizontal',ticks=[0])
	if k_ == 2:
		cax.text(0.5,-2.25,'Cycle number',ha='center',va='center')
		# cax.text(0.5,2.5,'Elapsed time [$t$] (hrs)',ha='center',va='center')
	chb.ax.set_xticklabels(str(k_+1))
	# cax.text(0,1.1,k_*Tc,ha='center')

# cax.text(1,1.1,(k_ + 1)*Tc,ha='center')


if issave:
	SAVE_FILE = 'JGLAC_Fig07_v1.5_Crossplots_%ddpi.%s'%(DPI,FMT.lower())
	plt.savefig(os.path.join(ODIR,SAVE_FILE),dpi=DPI,format=FMT.lower())



plt.show()




