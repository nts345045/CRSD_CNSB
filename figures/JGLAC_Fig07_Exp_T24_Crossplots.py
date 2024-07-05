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
ROOT = os.path.join('..')
DDIR = os.path.join(ROOT,'processed_data','cavities')
# Map Experimental Data
T24_CM = os.path.join(DDIR,'experiment_T24_cavity_metrics.csv')
MOD_CM = os.path.join(DDIR,'modeled_values.csv')

# Map output directory
ODIR = os.path.join(ROOT,'results','figures')

### LOAD EXPERIMENTAL DATA ###
# Load Experimental Data
df_T24 = pd.read_csv(T24_CM,parse_dates=True,index_col=[0])
df_MOD = pd.read_csv(MOD_CM,parse_dates=True,index_col=[0])

### SET REFERENCES AND OFFSETS ###
t0_T24 = pd.Timestamp('2021-10-26T18:58')
D_tau = 91.76 #60.95 # [kPa] - amount to reduce \tau(t)

### SET PLOTTING PARAMETERS
cmaps = ['Blues_r','Purples_r','RdPu_r','Oranges_r','Greys_r']
issave = True
DPI = 200
FMT = 'PNG'
URC = (0.9,0.85)
ULC = (0.05,0.90)
PADXY = 0.05



### PLOTTING SECTION ###
fig = plt.figure(figsize=(7.5,7.5))
GS = fig.add_gridspec(ncols=2,nrows=2,hspace=0,wspace=0)
axs = [fig.add_subplot(GS[0,0]),fig.add_subplot(GS[0,1]),\
	   fig.add_subplot(GS[1,0]),fig.add_subplot(GS[1,1])]

### (a) Plot \tau(N(t)) ###
XI = df_T24['N kPa'].values
YI = df_T24['T kPa'].values - D_tau
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
axs[0].set_ylabel('Shear Stress [$\\tau$] (kPa)')#,rotation=270,labelpad=15)
axs[0].set_xlabel('Effective Pressure [$N$] (kPa)')

axs[0].arrow(325,98,241-325,75.5-98,head_width=2,width=0.1,head_length=10,fc='k')
axs[0].text(271,89,'$-\partial_t N$',rotation=45,ha='center',va='center')
axs[0].arrow(252,28,389-252,45-28,head_width=2,width=0.1,head_length=10,fc='k')
axs[0].text(334,34,'$+\partial_t N$',rotation=30,ha='center',va='center')

### (b) Plot \tau(S(t)) ###
XI = df_T24['S tot'].values
YI = df_T24['T kPa'].values - D_tau
II = df_T24.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))
# Observed
plot_cycles(axs[1],XI,YI,II,\
			t0_T24,cmaps,ncycles=5,\
			T=pd.Timedelta(24,unit='hour'),zorder=10)
# Modeled values
axs[1].plot(df_MOD['Stot'].values,df_MOD['T_Pa'].values*1e-3,\
			  'r--',zorder=5)

axs[1].set_ylabel('Shear Stress $\\tau$ [kPa]',rotation=270,labelpad=15)
# axs[1].set_xlabel('$\\tau(t)$ [kPa]')
axs[1].set_xlim(xlims)
axs[1].set_ylim(ylims)
axs[1].yaxis.set_label_position('right')
axs[1].yaxis.set_ticks_position('right')
axs[1].xaxis.set_visible(False)

# Put in \partial_t N(t) annotations
axs[1].text(.23,60,'$-\\partial_t N$',rotation=60)
axs[1].text(.191,78.5,'$+\\partial_t N$',rotation=70,ha='center',va='center')



### (c) Plot T6 \Delta \mu(N(t)) ###
XI = df_T24['N kPa'].values
mu_obs = (df_T24['T kPa'].values - D_tau)/XI
# mu0_obs = np.mean(mu_obs[df_T24.index < t0_T24])
YI = mu_obs #- mu0_obs
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
Dmu_calc = mu_calc - mu_calc[0]
axs[2].plot(df_MOD.index.values*1e-3,mu_calc,\
			  'r--',zorder=5)
axs[2].set_ylabel('Drag [$\\mu$] ( - )')#,rotation=270,labelpad=15)
axs[2].set_xlabel('Effective Pressure [$N$] (kPa)')
axs[2].set_xlim(xlims)
axs[2].set_ylim(ylims)


### (d) Plot T6 \mu(S(t)) ###
XI = df_T24['S tot'].values
mu_obs = (df_T24['T kPa'].values - D_tau)/df_T24['N kPa'].values
YI = mu_obs
II = df_T24.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))
# Observed
plot_cycles(axs[3],XI,YI,II,\
			t0_T24,cmaps,ncycles=5,\
			T=pd.Timedelta(24,unit='hour'),zorder=10)
# Modeled values
mu_calc = df_MOD['T_Pa'].values/df_MOD.index.values
Dmu_calc = mu_calc - mu_calc[0]
axs[3].plot(df_MOD['Stot'].values,Dmu_calc,\
			  'r--',zorder=5)

axs[3].set_ylabel('Drag $\\mu$ [ - ]',rotation=270,labelpad=15)
axs[3].set_xlabel('Scaled Contact Length [$S$] ( - )')
axs[3].set_xlim(xlims)
axs[3].set_ylim(ylims)



### FORMATTING ###


lblkw = {'fontsize':14,'fontweight':'extra bold','fontstyle':'italic',\
		 'ha':'center','va':'center'}

for i_,lbl_ in enumerate(['a','b','c','d']):
	xlims = axs[i_].get_xlim()
	ylims = axs[i_].get_ylim()

	if i_%2 == 1:
		axs[i_].yaxis.set_label_position('right')
		axs[i_].yaxis.set_ticks_position('right')
		# axs[i_].set_xlim([0.1875,xlims[1]])
		# xlims = axs[i_].get_xlim()

	if i_%2 == 0:
		axs[i_].set_xlim([xlims[0],525])
		xlims = axs[i_].get_xlim()

	axs[i_].text(xlims[0] + (xlims[1] - xlims[0])*0.05,\
				 ylims[0] + (ylims[1] - ylims[0])*0.95,\
						 lbl_,**lblkw)

### COLORBAR PLOTTING ###

Tc = 24
cbar_placement = 'top'
# Create timing colorbar
for k_ in range(5):
	if cbar_placement.lower() == 'bottom':
		cax = fig.add_axes([.15 + (.70/5)*k_,.045,.70/5,.015])
	elif cbar_placement.lower() == 'top':
		cax = fig.add_axes([.15 + (.70/5)*k_,1 - .09,.70/5,.015])
	chb = plt.colorbar(chs[k_],cax=cax,orientation='horizontal',ticks=[0.99])
	if k_ == 2:
		if cbar_placement.lower() == 'bottom':
			cax.text(0.5,-2.25,'Cycle Number',ha='center',va='center')
		elif cbar_placement.lower() == 'top':
			cax.text(0.5,2.5,'Cycle Number',ha='center',va='center')
	chb.ax.set_xticklabels(str(k_+1))


if issave:
	SAVE_FILE = 'JGLAC_Fig07_T24_Crossplots_%ddpi.%s'%(DPI,FMT.lower())
	plt.savefig(os.path.join(ODIR,SAVE_FILE),dpi=DPI,format=FMT.lower())



plt.show()




