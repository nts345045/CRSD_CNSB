"""
:module: JGLAC_Fig04.py
:version: 1 - Revision for JOG-2024-0083 (Journal of Glaciology)
:short ref: Stevens and others - Experimental constraints on transient glacier slip with ice-bed separation
:Figure #: 4
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: 
	Plot cavity geometry measurements and comparison with instrument-inferred cavity geometry parameters
TODO: Clean up extraneous (commented-out) code
"""

import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.model.lliboutry_kamb_model as lkm

def main(args):

	# Observed Cavity Geometry DataData
	T24_SM = os.path.join(args.gpath,'Postprocessed_Cavity_Geometries.csv')
	# Processed Cavity Geometry Data
	T24_CM = os.path.join(args.gpath,'EX_T24_cavity_metrics.csv')
	# Steady-State Modelled Values
	MOD_CM = os.path.join(args.mpath,'cavity_geometry_mapping_values.csv')

	### LOAD EXPERIMENTAL DATA ###
	# Load Experimental Data
	df_OBS = pd.read_csv(T24_SM)
	df_OBS.index = pd.to_datetime(df_OBS.Epoch_UTC, unit='s')
	df_COR = pd.read_csv(T24_CM)
	df_COR.index = pd.to_datetime(df_COR.Epoch_UTC, unit='s')
	df_MOD = pd.read_csv(MOD_CM)
	df_MOD.index = df_MOD['N_Pa'].values

	### REFERENCE VALUES ###
	# Start of oscillation experiment
	t0_T24 = pd.Timestamp('2021-10-26T18:58')
	# Outer Wall bed geometry [in mm]
	aa_ow = 39.
	lbda_ow = 600.*np.pi*0.25 # 2 pi r / k


	# Photo-derived values
	# Stoss Side
	Sstobs = df_OBS['Stoss dX POR mm'].values/(-lbda_ow)
	Rstobs = 1 + df_OBS['Stoss dY POR mm'].values/(2.*aa_ow)
	# Lee Side
	Sleeobs = df_OBS['Lee dX PRO mm'].values/(lbda_ow)
	Rleeobs = 1 + df_OBS['Lee dY PRO mm'].values/(2.*aa_ow)
	# Total / Average
	Sobs = (df_OBS['Lee dX PRO mm'].values - df_OBS['Stoss dX POR mm'].values)/lbda_ow
	Robs = 1 - np.abs(df_OBS['Lee dY PRO mm'].values + df_OBS['Stoss dY POR mm'].values)/(4.*aa_ow)


	# Get Operational Range of Effective Pressures for Model
	IND = (df_COR['N kPa'].max()*1e3 + 1e3 >= df_MOD.index) &\
		(df_MOD.index >= df_COR['N kPa'].min()*1e3 - 1e3)

	# Model Cavity Roof Profiles
	gxm = lkm.calc_profiles(df_COR['N kPa'].min()*1e3)['gx']
	gxu = lkm.calc_profiles(df_COR['N kPa'].mean()*1e3)['gx']
	gxM = lkm.calc_profiles(df_COR['N kPa'].max()*1e3)['gx']

	# Model scaled bed profile
	Xbed = np.linspace(-.3,.3,101)
	Ybed = 0.5 + 0.5*np.cos(2*np.pi*Xbed)

	# GS1 = fig.add_gridspec(3,12,hspace=0)
	# GS2 = fig.add_gridspec(3,45,wspace=0)
	# axs = [fig.add_subplot(GS1[2,:10]),fig.add_subplot(GS1[1,:10]),\
	# 	fig.add_subplot(GS2[0,:12]),fig.add_subplot(GS2[0,12:43])]


	# Initalize Figure
	fig = plt.figure(figsize=(7.5,9))

	gs = fig.add_gridspec(3,12, hspace=0)
	axes = [fig.add_subplot(gs[0,:]),
		 	fig.add_subplot(gs[1,1:10]),
			fig.add_subplot(gs[2,1:10])]
	

	# SUBPLOT A: SPATIAL DISTRIBUTION OF PICKS 
	#######################
	# PLOT & LABEL OBSTACLE
	axes[0].fill_between(Xbed,np.ones(len(Xbed))*np.min(Ybed),Ybed,color='k',zorder=2)
	# Anatomy
	axes[0].text(0.15,.47,'Bed',color='w',ha='center',va='center', fontsize=14)
	axes[0].text(-.235, .45,'Stoss\nSide',color='w')
	axes[0].text(.1, .75, 'Lee\nSide', color='w')
	axes[0].text(0,1.07,'Bed Crest',color='w',ha='center',va='center')

	##############################
	# PLOT & ANNOTATE CAVITY ROOFS
	# Plot ice volumes
	axes[0].fill_between(np.linspace(-1,1,len(gxm)*2),np.r_[gxm,gxm]/(2.*25.3e-3),\
						np.ones(len(gxm)*2)*1.1,color='dodgerblue',zorder=1,alpha=0.33)
	axes[0].fill_between(np.linspace(-1,1,len(gxu)*2),np.r_[gxu,gxu]/(2.*25.3e-3),\
						np.ones(len(gxu)*2)*1.1,color='dodgerblue',zorder=1,alpha=0.33)
	axes[0].fill_between(np.linspace(-1,1,len(gxM)*2),np.r_[gxM,gxM]/(2.*25.3e-3),\
						np.ones(len(gxM)*2)*1.1,color='dodgerblue',zorder=1,alpha=0.33)
	
	# Plot lines at cavity roof edges
	axes[0].plot(np.linspace(0,1,len(gxm)),gxm/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
	axes[0].plot(np.linspace(0,1,len(gxu)),gxu/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
	axes[0].plot(np.linspace(0,1,len(gxM)),gxM/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
	axes[0].plot(np.linspace(-1,0,len(gxm)),gxm/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
	axes[0].plot(np.linspace(-1,0,len(gxu)),gxu/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
	axes[0].plot(np.linspace(-1,0,len(gxM)),gxM/(2.*25.3e-3),'-',zorder=1,color='dodgerblue')
	
	# Annotations
	# Cavity Labels
	axes[0].text(.133,0.875,'Cavity',va='center')
	axes[0].text(-.225,.93,'Largest Modeled Cavity',
			  ha='center',va='center',color='blue',
			  rotation=-3, fontsize=8)
	axes[0].text(-.24,.75, 'Mean Modeled Cavity',
			  ha='center',va='center',color='blue',
			  rotation=-7.5, fontsize=8)
	axes[0].text(-.2925,.54,'Smallest\nModeled\nCavity',
			  ha='left',va='center',color='blue',
			  rotation=0,fontsize=8)
	# Sliding Direction Label
	axes[0].arrow(-.28,1.05,.1,0,ec='w',fc='w',zorder=10,\
				width=.005,head_width=0.05,head_length=0.025)
	axes[0].text(-0.23,1.01,'Sliding Direction',color='w',ha='center',va='center')
	axes[0].text(-.1,1.,'Ice',ha='center',va='center', fontsize=14)

	#############################################
	# RANGE OF MODELED DETACHMENT POINT POSITIONS
	# Stoss
	axes[0].plot(-df_MOD[IND]['Sstoss'],df_MOD[IND]['Rstoss'],'r:',linewidth=2,zorder=4)
	# Lee
	axes[0].plot(df_MOD[IND]['Slee'],df_MOD[IND]['Rlee'],'r:',linewidth=2, zorder=4)
	# Minimum modeled contact area
	cIND = (Xbed >= -df_MOD[IND]['Sstoss'].min()) & (Xbed <= df_MOD[IND]['Slee'].min())
	axes[0].plot(Xbed[cIND],Ybed[cIND],'r-',linewidth=2)
	# Annotations

	axes[0].text(-.06,0.89,'Modeled Contact\nAreas',ha='center',va='center',color='r',\
				rotation=17)
	##################################
	# OVERLAY PHOTO-DERIVED POINT DATA
	# Photo-Derived Stoss Picks
	axes[0].scatter(-Sstobs,Rstobs,s=12,c=(df_OBS.index - t0_T24).total_seconds(),\
				zorder=5,alpha=0.75)
	# Photo-Derived Lee Picks
	axes[0].scatter(Sleeobs,Rleeobs,s=12,c=(df_OBS.index - t0_T24).total_seconds(),\
				zorder=5,alpha=0.75)
	# Annotate measures
	# Point Measurement Labels
	axes[0].text(0.018,1.015,'Detachment Points',color='w',ha='left')
	axes[0].text(-.225,.78,'Reattachment\nPoints',color='w',ha='left')
	REF = -9
	# R^{photo}
	axes[0].plot([0,0],[.405,np.mean([Rstobs[REF],Rleeobs[REF]])],'w--')
	axes[0].text(0.001, 0.7, '$R^{photo}$', color='w', rotation=270)
	# S^{photo}
	axes[0].plot([-Sstobs[REF], Sleeobs[REF]], [Rstobs[REF]]*2, 'w--')
	axes[0].text(np.mean([-Sstobs[REF],Sleeobs[REF]]), Rstobs[REF]+0.02, '$S^{photo}$', color='w', ha='center')

	# R_{stoss}^{photo}
	axes[0].plot([-Sstobs[REF]]*2, [.405, Rstobs[REF]], 'w:')
	axes[0].text(-Sstobs[REF]+0.001, 0.5, '$R_{stoss}^{photo}$',rotation=270, color='w')
	# S_{stoss}^{photo}
	axes[0].plot([-Sstobs[REF], 0], [0.45]*2,'w:')
	axes[0].text(-0.5*Sstobs[REF], 0.47,'$S_{stoss}^{photo}$', color='w', ha='center')

	# R_{lee}^{photo}
	axes[0].plot([Sleeobs[REF]]*2, [.405, Rleeobs[REF]], 'w:')
	axes[0].text(Sleeobs[REF]+0.001, 0.5, '$R_{lee}^{photo}$',rotation=270, color='w')
	# S_{lee}^{photo}
	axes[0].plot([0, Sleeobs[REF]], [np.mean([Rstobs[REF],Rleeobs[REF]])]*2, 'w:')
	axes[0].text(0.5*Sleeobs[REF], np.mean([Rstobs[REF],Rleeobs[REF]])+0.02, '$S_{lee}^{photo}$', color='w', ha='center')

	# FINAL FORMATTING
	axes[0].set_xlim([-.295,.195])
	axes[0].set_ylim([.405,1.1])
	axes[0].set_yticks([])
	axes[0].get_yaxis().set_visible(False)

	axes[0].text(0.18,1.05,'a', fontsize=14, fontweight='extra bold',fontstyle='italic', color='k')

	#####################
	# TIME-SERIES PLOTS

	XLIMS = [-12, 24*5 + 3]
	XTICKS = np.arange(-12,24*5+12,12)
	xticks = np.arange(-12,24*5+12,6)
	# SUBPLOT B: CAVITY HEIGHT 

	# Average R 
	# PHOTO
	axes[1].plot((df_OBS.index - t0_T24).total_seconds()/3600,Robs,\
				'ko-',ms=6,label='$R^{photo}$',zorder=3)
	# LVDT
	axes[1].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Robs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='o',zorder=4)
	# MODEL
	Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Rmea'].values)
	axes[1].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r-',label='$R^{calc}$',zorder=1)
	
	# Corrected LVDT
	axes[1].plot((df_COR.index - t0_T24).total_seconds()/3600,df_COR['R mea'].values,\
				'b-',lw=3,ms=4,zorder=2,label='$R^{LVDT}$')

	#### Stoss R
	# PHOTO
	axes[1].plot((df_OBS.index - t0_T24).total_seconds()/3600,Rstobs,\
				'kv-',ms=6,label='$R_{stoss}^{photo}$',zorder=3)
	
	axes[1].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Rstobs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='v',zorder=4)
	# MODEL
	Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Rstoss'].values)
	axes[1].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r--',label='$R^{calc}_{stoss}$',zorder=1)

	#### Lee R
	# PHOTO
	axes[1].plot((df_OBS.index - t0_T24).total_seconds()/3600,Rleeobs,\
				'ks-',ms=6,label='$R_{lee}^{photo}$',zorder=3)
	axes[1].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Rleeobs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='s',zorder=4)
	# MODEL
	Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Rlee'].values)
	axes[1].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r:',label='$R^{calc}_{lee}$',lw=2,zorder=1)



	# SUBPLOT C: CONTACT LENGTH
	## Total Contact Length
	# PHOTO
	axes[2].plot((df_OBS.index - t0_T24).total_seconds()/3600,Sobs,\
				'ko-',ms=6,label='$S^{photo}$',zorder=3)
	axes[2].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Sobs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='o',zorder=4)
	# MODEL
	Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Stot'].values)
	axes[2].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r-',zorder=1,lw=2, label='$S^{calc}$')

	# LVDT
	axes[2].plot((df_COR.index - t0_T24).total_seconds()/3600,df_COR['S tot'].values,\
				'b-',lw=3,ms=4,zorder=2,label='$S^{LVDT}$')
	

	# Stoss Contact Length
	# PHOTO
	axes[2].plot((df_OBS.index - t0_T24).total_seconds()/3600,Sstobs,\
				'kv-',ms=6,label='$S_{stoss}$',zorder=3)
	axes[2].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Sstobs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='v',zorder=4)
	# MODEL
	Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Sstoss'].values)
	axes[2].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r--',zorder=1,lw=1.5, label='$S^{calc}_{stoss}$')

	# Lee Contact Length
	# PHOTO
	axes[2].plot((df_OBS.index - t0_T24).total_seconds()/3600,Sleeobs,\
				'ks-',ms=6,label='$S_{lee}$',zorder=3)
	axes[2].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Sleeobs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='s',zorder=4)
	# MODEL
	Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Slee'].values)
	axes[2].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r:',zorder=1,lw=1.5, label='$S^{calc}_{lee}$')


	#### TIMESERIES FIXINS'
	for _e in range(1,3):
		ylims = axes[_e].get_ylim()
		axes[_e].plot([(df_OBS.index[REF] - t0_T24).total_seconds()/3600]*2, ylims, 'm-', alpha=0.33, zorder=1)
		axes[_e].set_ylim(ylims)
		axes[_e].legend(bbox_to_anchor=(1.,1))
		axes[_e].set_xticks(np.arange(-12,132,12))
		axes[_e].set_xticks(np.arange(-6,132,6), minor=True)
		axes[_e].grid(True, which='both',linestyle=':')

		axes[_e].set_xlim(XLIMS)

	axes[1].set_xticks([])
	axes[1].set_ylabel('Height Fraction [$R$] ( - )')
	axes[2].set_ylabel('Contact Fraction [$S$] ( - )')
	axes[2].set_xlabel('Elapsed Time During Exp. T24 (hr)')
	axes[1].text(115, 0.45, 'b', fontsize=14, fontweight='extra bold', fontstyle='italic')
	axes[2].text(115, 0.255, 'c', fontsize=14, fontweight='extra bold', fontstyle='italic')

	if not args.render_only:
		if args.dpi == 'figure':
			dpi = 'figure'
		else:
			try:
				dpi = int(args.dpi)

			except:
				dpi = 'figure'
		if dpi == 'figure':
			savename = os.path.join(args.output_path, f'JGLAC_Fig04_fdpi.{args.format}')
		else:
			savename = os.path.join(args.output_path, f'JGLAC_Fig04_{dpi}dpi.{args.format}')
		if not os.path.exists(os.path.split(savename)[0]):
			os.makedirs(os.path.split(savename)[0])
		plt.savefig(savename, dpi=dpi, format=args.format)

	if args.show:
		plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		prog='JGLAC_Fig04.py',
		description='Observed cavity geometry features from experiment T24'
	)

	parser.add_argument(
		'-e',
		'--experiment_path',
		dest='epath',
		default=os.path.join('..','..','processed_data','cavities'),
		help='Path to the cleaned up cavity geometry observation outputs',
		type=str
	)
	parser.add_argument(
		'-g',
		'--geometry_path',
		dest='gpath',
		default=os.path.join('..','..','processed_data','cavities'),
		help='Path to the cleaned up cavity geometry observation outputs',
		type=str
	)
	
	parser.add_argument(
		'-m',
		'--model_path',
		dest='mpath',
		default=os.path.join('..','..','processed_data','cavities'),
		help='Path to the cleaned up cavity geometry observation outputs',
		type=str
	)
	
	


	parser.add_argument(
		'-o',
		'--output_path',
		action='store',
		dest='output_path',
		default=os.path.join('..','..','results','figures'),
		help='path for where to save the rendered figure',
		type=str
	)

	parser.add_argument(
		'-f',
		'-format',
		action='store',
		dest='format',
		default='png',
		choices=['png','pdf','svg'],
		help='the figure output format (e.g., *.png, *.pdf, *.svg) callable by :meth:`~matplotlib.pyplot.savefig`. Defaults to "png"',
		type=str
	)

	parser.add_argument(
		'-d',
		'--dpi',
		action='store',
		dest='dpi',
		default='figure',
		help='set the `dpi` argument for :meth:`~matplotlib.pyplot.savefig. Defaults to "figure"'
	)

	parser.add_argument(
		'-s',
		'--show',
		action='store_true',
		dest='show',
		help='if included, render the figure on the desktop in addition to saving to disk'
	)
	
	parser.add_argument(
		'-r',
		'--render_only',
		dest='render_only',
		action='store_true',
		help='including this flag skips saving to disk'
	)
	args = parser.parse_args()
	main(args)