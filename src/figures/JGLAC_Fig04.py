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
	axs = [fig.add_subplot(GS1[2,:10]),fig.add_subplot(GS1[1,:10]),\
		fig.add_subplot(GS2[0,:12]),fig.add_subplot(GS2[0,12:43])]

	# Get Operational Range for Model
	IND = (df_COR['N kPa'].max()*1e3 + 1e3 >= df_MOD.index) &\
		(df_MOD.index >= df_COR['N kPa'].min()*1e3 - 1e3)

	axs[0].plot((df_OBS.index - t0_T24).total_seconds()/3600,Sobs,\
				'ko-',ms=6,label='$S$',zorder=3)
	axs[0].plot((df_OBS.index - t0_T24).total_seconds()/3600,Sstobs,\
				'kv-',ms=6,label='$S_{stoss}$',zorder=3)
	axs[0].plot((df_OBS.index - t0_T24).total_seconds()/3600,Sleeobs,\
				'ks-',ms=6,label='$S_{lee}$',zorder=3)

	# Plot color masking for association with Figs 4c-d
	axs[0].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Sobs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='o',zorder=4)
	axs[0].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Sstobs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='v',zorder=4)
	axs[0].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Sleeobs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='s',zorder=4)

	# Corrected LVDT
	axs[0].plot((df_COR.index - t0_T24).total_seconds()/3600,df_COR['S tot'].values,\
				'b-',lw=3,ms=4,zorder=2,label='$S^{LVDT}$')
	# Modeled
	mod_tuples = [('Stot','r-',2, '$S^{calc}$'),
			   	  ('Sstoss','r--',1.5, '$S^{calc}_{stoss}$'),
				  ('Slee','r:',1.5, '$S^{calc}_{lee}$')]
	for fld_,fmt_,lw_,lbl_ in mod_tuples:
		Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD[fld_].values)
		axs[0].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,fmt_,zorder=1,lw=lw_, label=lbl_)

	# Axis formatting
	axs[0].set_xticks(np.arange(0,132,12))
	axs[0].grid(axis='x',linestyle=':')
	axs[0].set_ylabel('Contact Fraction ( - )')
	axs[0].set_xticks(np.arange(0,132,12))
	axs[0].grid(axis='x',linestyle=':')
	# Legend
	axs[0].legend(bbox_to_anchor=(1., 1))

	### (a) TIMESERIES OF R(t)
	# Photogrametry 
	axs[1].plot((df_OBS.index - t0_T24).total_seconds()/3600,Robs,\
				'ko-',ms=6,label='$R$',zorder=3)
	axs[1].plot((df_OBS.index - t0_T24).total_seconds()/3600,Rstobs,\
				'kv-',ms=6,label='$R_{stoss}$',zorder=3)
	axs[1].plot((df_OBS.index - t0_T24).total_seconds()/3600,Rleeobs,\
				'ks-',ms=6,label='$R_{lee}$',zorder=3)

	axs[1].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Robs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='o',zorder=4)
	axs[1].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Rstobs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='v',zorder=4)
	axs[1].scatter((df_OBS.index - t0_T24).total_seconds()/3600,Rleeobs,\
				s=5,c=(df_OBS.index - t0_T24).total_seconds()/3600,\
				marker='s',zorder=4)

	# Corrected LVDT
	axs[1].plot((df_COR.index - t0_T24).total_seconds()/3600,df_COR['R mea'].values,\
				'b-',lw=3,ms=4,zorder=2,label='$R^{LVDT}$')
	# Modeled
	# for fld_,fmt_ in [('Rstoss','r--'),('Rmea','r-'),('Rlee','r:')]:
	Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Rmea'].values)
	axs[1].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r-',label='$R^{calc}$',zorder=1)
	Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Rstoss'].values)
	axs[1].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r--',label='$R^{calc}_{stoss}$',zorder=1)
	Y_ = np.interp(df_COR['N kPa'].values*1e3,df_MOD.index.values,df_MOD['Rlee'].values)
	axs[1].plot((df_COR.index - t0_T24).total_seconds()/3600,Y_,'r:',label='$R^{calc}_{lee}$',lw=2,zorder=1)

	axs[1].legend(bbox_to_anchor=(1.,1))

	axs[1].set_xticks(np.arange(0,132,12))
	axs[1].grid(axis='x',linestyle=':')
	axs[1].set_ylabel('Height Fraction ( - )')
	axs[0].set_xticklabels([])
	ylims = axs[1].get_ylim()

	axs[1].set_xlabel('Elapsed Time from Start of Experiment T24 (hr)')

	TI = np.linspace((df_OBS.index[0] - t0_T24).total_seconds()/3600,\
					(df_OBS.index[-1] - t0_T24).total_seconds()/3600,\
					201)
	axs[1].scatter(TI,0.01*(ylims[1] - ylims[0]) + ylims[0]*np.ones(len(TI)),s=9,c=TI,marker='s')
	axs[1].set_ylim(ylims)

	for i_ in range(2):
		axs[i_].set_xlim([(df_COR.index[0] - t0_T24).total_seconds()/3600,\
						(df_COR.index[-1] - t0_T24).total_seconds()/3600 + 2])



	# ### (c) CROSSPLOT OF S(\bar{R}(t))
	axs[2].scatter(Sobs,Robs,s=9,c=(df_OBS.index - t0_T24).total_seconds(),\
				zorder=2)
	axs[2].plot(df_MOD[IND]['Stot'].values,df_MOD[IND]['Rmea'].values,'r-',zorder=1)

	axs[2].set_ylabel('Height Fraction [$R$] ( - )')#,rotation=270,labelpad=15,zorder=10)
	axs[2].set_xlabel('Contact Fraction [$S$] ( - )')


	# ### (d) SCALED PICK LOCATIONS
	# # Plot Stoss Picks
	axs[3].scatter(-Sstobs,Rstobs,s=9,c=(df_OBS.index - t0_T24).total_seconds(),\
				zorder=5,alpha=0.75)
	# # Plot Lee Picks
	axs[3].scatter(Sleeobs,Rleeobs,s=9,c=(df_OBS.index - t0_T24).total_seconds(),\
				zorder=5,alpha=0.75)
	# # Plot Bed Profile
	Xbed = np.linspace(-.3,.3,101)
	Ybed = 0.5 + 0.5*np.cos(2*np.pi*Xbed)
	axs[3].fill_between(Xbed,np.ones(len(Xbed))*np.min(Ybed),Ybed,color='k',zorder=2)

	# Plot Cavity Profile
	gxm = lkm.calc_profiles(df_COR['N kPa'].min()*1e3)['gx']
	gxu = lkm.calc_profiles(df_COR['N kPa'].mean()*1e3)['gx']
	gxM = lkm.calc_profiles(df_COR['N kPa'].max()*1e3)['gx']
	# print(f"{df_COR['N kPa'].min()} - {df_COR['N kPa'].mean()} - {df_COR['N kPa'].max()}")
	xlims = axs[3].get_xlim()

	# # Plot Cavity Roof Positions
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
	# # Plot Contact range

	axs[3].plot(-df_MOD[IND]['Sstoss'],df_MOD[IND]['Rstoss'],'r--',zorder=4)
	axs[3].plot(df_MOD[IND]['Slee'],df_MOD[IND]['Rlee'],'r.',zorder=4)
	cIND = (Xbed >= -df_MOD[IND]['Sstoss'].min()) & (Xbed <= df_MOD[IND]['Slee'].min())
	axs[3].plot(Xbed[cIND],Ybed[cIND],'r-')


	# # Label Axes
	axs[3].set_ylabel('Scaled Y-Position\n[$y (2a)^{-1}$] ( - )',rotation=270,labelpad=30)
	axs[3].set_xlabel('Scaled X-Position [$x \\lambda ^{-1}$] ( - )')
	axs[3].set_xlim([-.295,.195])
	axs[3].set_ylim([.405,1.1])

	# # Annotations in (c)
	axs[3].arrow(-.2,.45,.3,0,ec='w',fc='w',zorder=10,\
				width=.005,head_width=0.05,head_length=0.05)
	axs[3].text(-.2, .5,'Stoss\nSide',color='w')
	axs[3].text(.1, .5, 'Lee\nSide', color='w')
	axs[3].text(-0.05,0.5,'Rotation Direction',color='w',ha='center',va='center')
	axs[3].text(0.04,0.9,'$p_{d}$',color='w')
	axs[3].text(-.16,.66,'$p_{r}$',color='w')
	axs[3].text(0,0.90,'Crest',color='w',ha='center',va='center')
	axs[3].text(0,.65,'Bed',color='w',ha='center',va='center')
	axs[3].text(-.2,0.975,'Ice',ha='center',va='center')
	axs[3].text(.133,0.85,'Cavity',va='center')
	axs[3].text(-.06,1.03,'Contact',ha='center',va='center',color='r',\
				rotation=12.5)
	axs[3].text(-.228,.820,'Modeled\nCavity Roofs',\
				ha='center',va='center',color='blue',\
				rotation=-7.5)


	axs[3].yaxis.set_label_position('right')
	axs[3].yaxis.set_ticks_position('right')


	lblkw = {'fontsize':14,'fontweight':'extra bold','fontstyle':'italic'}
	axs[1].text(-3,0.5,'b',ha='right',va='bottom',**lblkw)
	axs[0].text(-3,0.25,'a',ha='right',va='center',**lblkw)
	axs[3].text(0.1685,1.025,'d',ha='center',va='center',**lblkw)
	axs[2].text(.26,.95,'c',ha='center',va='center',**lblkw)



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