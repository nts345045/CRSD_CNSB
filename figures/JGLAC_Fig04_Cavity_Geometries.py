"""
JGLAC_Fig05_v1.4_Cavity_Geometries.py


"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.model.lliboutry_kamb_model as lkm

# plt.rc('text',usetex=True)

# Map Data
ROOT = os.path.join('..')
DDIR = os.path.join(ROOT,'processed_data','cavities')
# Map Experimental Data
# Observed Cavity Geometry DataData
T24_SM = os.path.join(DDIR,'Postprocessed_Contact_Geometries.csv')
# Processed Cavity Geometry Data
T24_CM = os.path.join(DDIR,'experiment_T24_cavity_metrics.csv')
# Steady-State Modelled Values
MOD_CM = os.path.join(DDIR,'modeled_values.csv')

# Map output directory
ODIR = os.path.join(ROOT,'results','figures')
issave = True
DPI = 200
FMT = 'PNG'


### LOAD EXPERIMENTAL DATA ###
# Load Experimental Data
df_OBS = pd.read_csv(T24_SM,parse_dates=True,index_col=[0])
df_COR = pd.read_csv(T24_CM,parse_dates=True,index_col=[0])
df_MOD = pd.read_csv(MOD_CM,parse_dates=True,index_col=[0])


### REFERENCE VALUES ###
# Start of oscillation experiment
t0_T24 = pd.Timestamp('2021-10-26T13:58')
# Outer Wall bed geometry [in mm]
aa_ow = 39.
lbda_ow = 600.*np.pi*0.25 # 2 pi r / k


# Photogrametry observed values
# Stoss Side
Sstobs = df_OBS['Stoss dX POR mm'].values/(-lbda_ow)
Rstobs = 1 + df_OBS['Stoss dY POR mm'].values/(2.*aa_ow)
# Lee Side
Sleeobs = df_OBS['Lee dX PRO mm'].values/(lbda_ow)
Rleeobs = 1 + df_OBS['Lee dY PRO mm'].values/(2.*aa_ow)
# Total / Average
Sobs = (df_OBS['Lee dX PRO mm'].values - df_OBS['Stoss dX POR mm'].values)/lbda_ow
Robs = 1 - np.abs(df_OBS['Lee dY PRO mm'].values + df_OBS['Stoss dY POR mm'].values)/(4.*aa_ow)



# Initalize Figure
fig = plt.figure(figsize=(7.5,7.5))
GS1 = fig.add_gridspec(3,12,hspace=0)
GS2 = fig.add_gridspec(3,45,wspace=0)
axs = [fig.add_subplot(GS1[0,:10]),fig.add_subplot(GS1[1,:10]),\
	   fig.add_subplot(GS2[2,:12]),fig.add_subplot(GS2[2,12:43])]

# Get Operational Range for Model
IND = (df_COR['N kPa'].max()*1e3 + 1e3 >= df_MOD.index) &\
	  (df_MOD.index >= df_COR['N kPa'].min()*1e3 - 1e3)

### (a) TIMESERIES OF R(t)
# Photogrametry 
axs[0].plot((df_OBS.index - t0_T24).total_seconds()/3600,Robs,\
			'ko-',ms=6,label='a) Obs. Mean\nb) Obs. Total',zorder=3)
axs[0].plot((df_OBS.index - t0_T24).total_seconds()/3600,Rstobs,\
			'kv-',ms=6,label='Obs. Stoss',zorder=3)
axs[0].plot((df_OBS.index - t0_T24).total_seconds()/3600,Rleeobs,\
			'ks-',ms=6,label='Obs. Lee',zorder=3)

axs[0].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Robs,\
			s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
			marker='o',zorder=4)
axs[0].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Rstobs,\
			s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
			marker='v',zorder=4)
axs[0].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Rleeobs,\
			s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
			marker='s',zorder=4)

# Corrected LVDT
axs[0].plot((df_COR.index - t0_T24).total_seconds()/3600,df_COR['R mea'].values,\
			'b-',lw=3,ms=4,zorder=2,label='LVDT-Based')
# Modeled
# for fld_,fmt_ in [('Rstoss','r--'),('Rmea','r-'),('Rlee','r:')]:
Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Rmea'].values)
axs[0].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r-',label='a) Mod. Mean\nb) Mod. Total',zorder=1)
Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Rstoss'].values)
axs[0].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r--',label='Mod. Stoss',zorder=1)
Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Rlee'].values)
axs[0].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r:',label='Mod. Lee',lw=2,zorder=1)


axs[0].set_xticks(np.arange(0,132,12))
axs[0].grid(axis='x',linestyle=':')
# axs[0].legend(ncol=3)
axs[0].set_ylabel('Scaled Cavity Height\n[$R$] ( - )')

ylims = axs[0].get_ylim()
axs[0].xaxis.set_ticks_position('top')
axs[0].xaxis.set_label_position('top')
axs[0].set_xlabel('Elapsed time (hr)')

TI = np.linspace((df_OBS.index[0] - t0_T24).total_seconds()/3600,\
			     (df_OBS.index[-1] - t0_T24).total_seconds()/3600,\
			     201)
axs[0].scatter(TI,0.01*(ylims[1] - ylims[0]) + ylims[0]*np.ones(len(TI)),s=9,c=TI,marker='s')
axs[0].set_ylim(ylims)
axs[0].legend(bbox_to_anchor=(1.,0.45))




### (b) TIMESERIES OF S(t)
axs[1].plot((df_OBS.index - t0_T24).total_seconds()/3600,Sstobs,\
	        'kv-',ms=6,label='Stoss',zorder=3)
axs[1].plot((df_OBS.index - t0_T24).total_seconds()/3600,Sleeobs,\
	        'ks-',ms=6,label='Lee',zorder=3)
axs[1].plot((df_OBS.index - t0_T24).total_seconds()/3600,Sobs,\
	        'ko-',ms=6,label='Total',zorder=3)

axs[1].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Sobs,\
			s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
			marker='o',zorder=4)
axs[1].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Sstobs,\
			s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
			marker='v',zorder=4)
axs[1].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Sleeobs,\
			s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
			marker='s',zorder=4)

# Corrected LVDT
axs[1].plot((df_COR.index - t0_T24).total_seconds()/3600,df_COR['S tot'].values,\
			'b-',lw=3,ms=4,zorder=2,label='LVDT')

axs[1].set_xticks(np.arange(0,132,12))
axs[1].grid(axis='x',linestyle=':')
axs[1].set_ylabel('Scaled Contact Length\n[$S$] ( - )')
# axs[1].legend(ncol=3)
# Modeled
for fld_,fmt_,lw_ in [('Sstoss','r--',1.5),('Stot','r-',1.5),('Slee','r:',2)]:
	Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD[fld_].values)
	axs[1].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,fmt_,zorder=1,lw=lw_)

axs[1].set_xticks(np.arange(0,132,12))
axs[1].grid(axis='x',linestyle=':')
# axs[1].legend(ncol=3)
# axs[1].xaxis.set_visible(False)

# Slvdt = np.interp(df_COR['R mea'].values,df_MOD['Rmea'].values,df_MOD['Stot'].values,period=24)
# axs[1].plot((df_COR.index - t0_T24).total_seconds()/3600,Slvdt,'b-',zorder=2)


for i_ in range(2):
	axs[i_].set_xlim([(df_COR.index[0] - t0_T24).total_seconds()/3600,\
					 (df_COR.index[-1] - t0_T24).total_seconds()/3600])



### (c) CROSSPLOT OF S(\bar{R}(t))
axs[2].scatter(Sobs,Robs,s=9,c=(df_OBS.index - t0_T24).total_seconds(),\
			   zorder=2)
			   # vmin=0,vmax=5*24*3600,zorder=2)
axs[2].plot(df_MOD[IND]['Stot'].values,df_MOD[IND]['Rmea'].values,'r-',zorder=1)

axs[2].set_ylabel('Scaled Cavity Height\n[$R$] ( - )')#,rotation=270,labelpad=15,zorder=10)
axs[2].set_xlabel('Scaled Contact Length\n[$S$] ( - )')


### (d) SCALED PICK LOCATIONS
# Plot Stoss Picks
axs[3].scatter(-Sstobs,Rstobs,s=9,c=(df_OBS.index - t0_T24).total_seconds(),\
			   zorder=5,alpha=0.75)
# Plot Lee Picks
axs[3].scatter(Sleeobs,Rleeobs,s=9,c=(df_OBS.index - t0_T24).total_seconds(),\
			   zorder=5,alpha=0.75)
# Plot Bed Profile
Xbed = np.linspace(-.3,.3,101)
Ybed = 0.5 + 0.5*np.cos(2*np.pi*Xbed)
axs[3].fill_between(Xbed,np.ones(len(Xbed))*np.min(Ybed),Ybed,color='k',zorder=2)

# Plot Cavity Profile
gxm = lkm.calc_profiles(df_COR['N kPa'].min()*1e3)['gx']
gxu = lkm.calc_profiles(df_COR['N kPa'].mean()*1e3)['gx']
gxM = lkm.calc_profiles(df_COR['N kPa'].max()*1e3)['gx']

xlims = axs[3].get_xlim()

# Plot Cavity Roof Positions
axs[3].fill_between(np.linspace(-1,1,len(gxm)*2),np.r_[gxm,gxm]/(2.*25.3e-3),\
					np.ones(len(gxm)*2)*1.1,color='dodgerblue',zorder=1,alpha=0.33)
axs[3].fill_between(np.linspace(-1,1,len(gxu)*2),np.r_[gxu,gxu]/(2.*25.3e-3),\
					np.ones(len(gxu)*2)*1.1,color='dodgerblue',zorder=1,alpha=0.33)
axs[3].fill_between(np.linspace(-1,1,len(gxM)*2),np.r_[gxM,gxM]/(2.*25.3e-3),\
					np.ones(len(gxM)*2)*1.1,color='dodgerblue',zorder=1,alpha=0.33)



axs[3].plot(np.linspace(0,1,len(gxm)),gxm/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
axs[3].plot(np.linspace(0,1,len(gxu)),gxu/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
axs[3].plot(np.linspace(0,1,len(gxM)),gxM/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
axs[3].plot(np.linspace(-1,0,len(gxm)),gxm/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
axs[3].plot(np.linspace(-1,0,len(gxu)),gxu/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
axs[3].plot(np.linspace(-1,0,len(gxM)),gxM/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
# Plot Contact range

axs[3].plot(-df_MOD[IND]['Sstoss'],df_MOD[IND]['Rstoss'],'r--',zorder=4)
axs[3].plot(df_MOD[IND]['Slee'],df_MOD[IND]['Rlee'],'r.',zorder=4)
cIND = (Xbed >= -df_MOD[IND]['Sstoss'].min()) & (Xbed <= df_MOD[IND]['Slee'].min())
axs[3].plot(Xbed[cIND],Ybed[cIND],'r-')


# Label Axes
axs[3].set_ylabel('Scaled Y-Position\n[$y (2a)^{-1}$] ( - )',rotation=270,labelpad=30)
axs[3].set_xlabel('Scaled X-Position [$x \\lambda ^{-1}$] ( - )')
axs[3].set_xlim([-.295,.195])
axs[3].set_ylim([.405,1.1])

# Annotations in (c)
axs[3].arrow(-.2,.45,.3,0,ec='w',fc='w',zorder=10,\
			 width=.005,head_width=0.05,head_length=0.05)
axs[3].text(-0.05,0.5,'Rotation Direction',color='w',ha='center',va='center')
axs[3].text(0.05,0.8,'Lee\n[$p_{d}$]',color='w')
axs[3].text(-.15,.65,'Stoss\n[$p_{r}$]',color='w')
axs[3].text(0,0.90,'Crest\n[$p_{c}$]',color='w',ha='center',va='center')
axs[3].text(0,.65,'Bed',color='w',ha='center',va='center')
axs[3].text(-.2,0.975,'Ice',ha='center',va='center')
axs[3].text(.133,0.85,'Cavity',va='center')
axs[3].text(-.06,1.03,'Contact',ha='center',va='center',color='r',\
			rotation=12.5)
axs[3].text(-.228,.833,'Modeled\nCavity Roofs',\
			ha='center',va='center',color='blue',\
			rotation=-7.5)


axs[3].yaxis.set_label_position('right')
axs[3].yaxis.set_ticks_position('right')
# axs[3].set_xlim([df_MOD[IND]['Stot'].min(),df_MOD[IND]['Stot'].max()])
# axs[3].set_ylim([df_MOD[IND]['Rmea'].min(),df_MOD[IND]['Rmea'].max()])


lblkw = {'fontsize':14,'fontweight':'extra bold','fontstyle':'italic'}
axs[0].text(0,0.5,'a',ha='right',va='bottom',**lblkw)
axs[1].text(0,0.25,'b',ha='right',va='center',**lblkw)
axs[3].text(0.1685,1.025,'d',ha='center',va='center',**lblkw)
axs[2].text(.26,.95,'c',ha='center',va='center',**lblkw)






if issave:
	OFILE = os.path.join(ODIR,'JGLAC_Fig04_Cavity_Geometries_%ddpi.%s'%(DPI,FMT.lower()))
	plt.savefig(OFILE,dpi=DPI,format=FMT.lower())


plt.show()