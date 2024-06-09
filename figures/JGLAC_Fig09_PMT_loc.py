"""
JGLAC_Fig09_PMT_loc.py

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Plotting Controls #
isplot = True
cmaps = ['Blues_r','Purples_r','RdPu_r','Oranges_r','Greys_r']
issave = True
DPI = 200
FMT = 'PNG'
URC = (0.9,0.85)
ULC = (0.05,0.90)
PADXY = 0.05

THETA_REF = -0.034 	# [\degree C] Experimental chamber temperature
DTHETA_REF = 0.01 	# [\degree C] Experimental chamber temperature control
D_tau = 0 			# [Pa] Offset value 
Otp = 273.15 		# [K] Pure water triple-point temperature
Ptp = 611.73 		# [Pa] Pure water triple-point pressure
yCC = 9.8e-8 		# [K/Pa] Clausius-Clapeyron parameter (value from Hooke, 2005)
Dtau_24 = 91.76 	# [kPa] Shear stress correction factor for Exp. T24 steady-state
Dtau_06 = 90.43 	# [kPa] Shear stress correction factor for Exp. T06 steady-state
def Opmt(N,Otp=Otp,Ptp=Ptp,yCC=yCC):
	"""
	Calculate pressure melting point of ice for a given pressure
	:: INPUTS ::
	:param N: Pressure(s) in [Pa]
	:param Otp: Triple-point temperature [Kelvin]
	:param Ptp: Triple-point pressure [Pa]
	:param yCC: Clausius-Clapeyron parameter [Kelvin/Pa]

	:: OUTPUT ::
	:return Opmt: Pressure melting temperature(s) [Kelvin]
	"""
	Opmt = Otp - yCC*(N - Ptp)
	return Opmt

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

# Map Directories
ROOT = os.path.join('..','..')
DDIR = os.path.join(ROOT,'processed','timeseries')

# Map Experimental Data Files
T24_DAT = os.path.join(DDIR,'S5_experiment_T24_cavity_metrics.csv')
T06_DAT = os.path.join(DDIR,'S5_experiment_T6_cavity_metrics.csv')
# Map output directory
ODIR = os.path.join(ROOT,'results','figures','manuscript')

# LOAD EXPERIMENTAL DATA #
df_24 = pd.read_csv(T24_DAT,parse_dates=True,index_col=[0])
df_06 = pd.read_csv(T06_DAT,parse_dates=True,index_col=[0])


# Define Reference time for T24 and T06
t0_T24 = pd.Timestamp('2021-10-26T13')
t0_T06 = pd.Timestamp('2021-11-1T11:09:15')


# CALCULATE SIGMA_LOC
o_loc_24 = df_24['N kPa'].values/df_24['S tot'].values
o_loc_06 = df_06['N kPa'].values/df_06['S tot'].values

# CALCULATE Opmt N(t)
O_Nt_24 = Opmt(df_24['N kPa'].values*1e3) - Otp
O_Nt_06 = Opmt(df_06['N kPa'].values*1e3) - Otp

# CALCULATE Opmt LOC
O_loc_24 = Opmt(o_loc_24*1e3) - Otp
O_loc_06 = Opmt(o_loc_06*1e3) - Otp

# CALCULATE mu
# Get steady-state corrected drags
tau_C24 = df_24['T kPa'].values - Dtau_24
tau_C06 = df_06['T kPa'].values - Dtau_06
# Calculate drag
mu_C24 = tau_C24/df_24['N kPa'].values
mu_C06 = tau_C06/df_06['N kPa'].values



# N0_24 = df_24[df_24.index > t0_T24 + pd.Timedelta(121,unit='hour')]['N kPa'].mean()
# mu_24 = (df_24['T kPa'].values - Tau0_24)/(df_24['N kPa'].values - N0_24)
# Dmu_24 = mu_24 - mu_24[-1] #(df_24['T kPa'].values[-1]/df_24['N kPa'].values[-1])
# # mu_06 = (df_06['T kPa'].values - df_06['T kPa'].values[0])/df_06['N kPa'].values
# # Dmu_06 = mu_06 - (df_06['T kPa'].values[-1]/df_06['N kPa'].values[-1])
# Tau0_06 = df_06[df_06.index > t0_T06 + pd.Timedelta(121,unit='hour')]['T kPa'].mean()
# mu_06 = (df_06['T kPa'].values - Tau0_06)/df_06['N kPa'].values
# Dmu_06 = mu_06 - mu_06[-1] #(df_24['T kPa'].values[-1]/df_24['N kPa'].values[-1])



fig = plt.figure(figsize=(7.5,7.5))
GS = fig.add_gridspec(ncols=2,nrows=2,wspace=0)

axs = [fig.add_subplot(GS[0,0]),fig.add_subplot(GS[0,1]),\
	   fig.add_subplot(GS[1,0]),fig.add_subplot(GS[1,1])]

# (a) \sigma_{loc} (t) & N(t) for T24
axs[0].plot((df_24.index - t0_T24).total_seconds()/3600/24,\
			  df_24['N kPa'].values*1e-3,'k-',label='Effective [$N$]')

axs[0].plot((df_24.index - t0_T24).total_seconds()/3600/24,\
			  o_loc_24*1e-3,'-',color='royalblue',label='Contact [$\\sigma_{loc}$]')
axs[0].legend(loc='upper right')
axs[0].set_xticks(np.arange(0,6,1))
axs[0].set_title("Experiment T24")
axs[0].set_ylabel('Pressure (MPa)')

# (b) \sigma_{loc} (t) & N(t) for T06
axs[1].plot((df_06.index - t0_T06).total_seconds()/3600/6,\
			  df_06['N kPa'].values*1e-3,'k-',label='Effective [$N$]')

axs[1].plot((df_06.index - t0_T06).total_seconds()/3600/6,\
			  o_loc_06*1e-3,'-',color='royalblue',label='Contact [$\\sigma_{loc}$]')
# axs[1].legend(loc='upper left')

ylim1 = axs[0].get_ylim()
axs[1].set_ylim(ylim1)
axs[1].yaxis.set_label_position('right')
axs[1].yaxis.set_ticks_position('right')
axs[1].set_ylabel('Pressure [MPa]',rotation=270,labelpad=15)
axs[1].set_title('Experiment T06')

for j_ in range(2):
	axs[j_].set_xticks(np.arange(0,6,1))
	axs[j_].set_xlim([-0.5,5.5])
	axs[j_].set_xlabel('Cycle [No.]')





### (c) Exp T24 Drag - Relative temperature Crosspolot ###
YI = mu_C24
XI = THETA_REF - O_loc_24
II = df_24.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))

chs = plot_cycles(axs[2],XI,YI,II,t0_T24,cmaps,\
				  ncycles=5,T=pd.Timedelta(24,unit='hour'),zorder=10)
axs[2].set_xlim(xlims)
axs[2].set_ylim(ylims)
axs[2].set_ylabel('Drag [$\\mu$] ( - )')
axs[2].set_xlabel('Change in PMT [$\\Delta \\Theta$] ($K$)')



### (d) Exp T06 Drag - Relative temperature Crosspolot ###
YI = mu_C06
XI = THETA_REF - O_loc_06
II = df_06.index
xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
		 np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
		 np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))

chs = plot_cycles(axs[3],XI,YI,II,t0_T06,cmaps,\
				  ncycles=5,T=pd.Timedelta(6,unit='hour'),zorder=10)
axs[3].set_xlim(xlims)
axs[3].set_ylim(ylims)
axs[3].set_ylabel('Drag [$\\mu$] ( - )',rotation=270,labelpad=15)
axs[3].set_xlabel('Change in PMT [$\\Delta \\Theta$] ($K$)')
axs[3].yaxis.set_label_position('right')
axs[3].yaxis.set_ticks_position('right')

lblkw = {'fontsize':14,'fontweight':'extra bold','fontstyle':'italic',\
		 'ha':'center','va':'center'}

for i_,lbl_ in enumerate(['a','b','c','d']):
	xlims = axs[i_].get_xlim()
	ylims = axs[i_].get_ylim()


	axs[i_].text(xlims[0] + (xlims[1] - xlims[0])*0.05,\
				 ylims[0] + (ylims[1] - ylims[0])*0.95,\
						 lbl_,**lblkw)


if issave:
	OFILE = os.path.join(ODIR,'JGLAC_Fig09_v1.6_PMT_loc_%ddpi.%s'%(DPI,FMT.lower()))
	plt.savefig(OFILE,dpi=DPI,format=FMT.lower())


plt.show()