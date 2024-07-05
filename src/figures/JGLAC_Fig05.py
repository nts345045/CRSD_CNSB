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
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(args):
	# # Map Directories
	# ROOT = os.path.join('..')
	# DDIR = os.path.join(ROOT,'processed_data')
	# Map Experimental Data Files
	T24_NT = os.path.join(args.input_path,'5_split_data','EX_T24-Pressure.csv')
	T24_LV = os.path.join(args.input_path,'6_lvdt_melt_corrected','EX_T24-LVDT-reduced.csv')
	T24_CM = os.path.join(args.input_path,'cavities','EX_T24_cavity_metrics.csv')
	T24_CP = os.path.join(args.input_path,'cavities','Postprocessed_Cavity_Geometries.csv')
	# Map output directory
	# ODIR = os.path.join(ROOT,'results','figures')

	# LOAD EXPERIMENTAL DATA #
	df_NT24 = pd.read_csv(T24_NT)
	df_Z24 = pd.read_csv(T24_LV)
	df_CM24 = pd.read_csv(T24_CM)
	df_CP24 = pd.read_csv(T24_CP)

	df_NT24.index = pd.to_datetime(df_NT24.Epoch_UTC,unit='s')
	df_Z24.index = pd.to_datetime(df_Z24.Epoch_UTC, unit='s')
	df_CM24.index = pd.to_datetime(df_CM24.Epoch_UTC, unit='s')
	df_CP24.index = pd.to_datetime(df_CP24.Epoch_UTC, unit='s')

	# Plotting Controls #
	# issave = True
	# DPI = 200
	# FMT = 'PNG'

	# Define Reference time for T24
	t0_T24 = pd.Timestamp('2021-10-26T18:58')
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

	# ylim = [-6,121]

	### CALCULATE OBSERVED DRAG ###
	# Observed Drag with \Delta \tau offset
	mu_obs = (df_NT24['Tau_kPa'].values - D_tauP)/df_NT24['Pe_kPa'].values
	df_mu_obs = pd.DataFrame({'mu_obs':mu_obs},index=df_NT24.index)
	# Get Reference \\mu(t) value for differencing
	mu0_obs = df_mu_obs[dtindex >= 121 ].mean().values
	# Get observed change in drag
	Dmu_obs = mu_obs - mu0_obs

	### CALCULATE SHIFTED DRAG ###
	mu_tP = (df_NT24['Tau_kPa'].values - D_tauP)/df_NT24['Pe_kPa'].values


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
	axs[0].plot(dtindex,df_NT24['Tau_kPa'] - D_tau,'k-',zorder=10)
	# Plot modeled values
	axs[0].plot(dtzindex,df_CM24['hat T kPa'].values ,'r--',zorder=5)
	axs[0].plot(dtindex,df_NT24['Tau_kPa'] - D_tauP,'b-',zorder=8)
	# Apply labels & formatting
	axs[0].set_ylabel('Shear Stress\n[$\\tau$] (kPa)')
	# axb.set_ylabel('$\\hat{\\tau}$(t) [kPa]',rotation=270,labelpad=15,color='red')
	axs[0].set_xticks(np.arange(0,132,12))
	axs[0].grid(axis='x',linestyle=':')
	axs[0].text(115,df_CM24['hat T kPa'].values[-1] + D_tauP/2,'$\\Delta \\tau$',fontsize=14,ha='center',va='center')
	axs[0].arrow(118,153,0,95.4 - 154,head_width=2,width=0.1,head_length=10,fc='k')


	# (b) PLOT \Delta \mu(t)
	# Plot observed values
	# axs[1].plot(dtindex,mu_obs,'k-',zorder=10,label='Obs.')
	# plot modeled values
	axs[1].plot(dtzindex,mu_calc ,'r--',zorder=5,label='Mod.')
	axs[1].plot(dtindex[np.isfinite(mu_tP)],mu_tP[np.isfinite(mu_tP)],'b-',zorder=5,label='$\\tau_{adj}$')
	# Apply labels & formatting
	axs[1].set_xticks(np.arange(0,132,12))
	axs[1].grid(axis='x',linestyle=':')
	axs[1].set_ylabel('Drag\n[$\\mu$] ( - )')
	# axc.set_ylabel('$\\Delta\\mu$ (t) [ - ]',rotation=270,labelpad=15,color='red')
	# axs[1].set_ylim([-0.11,0.22])
	# axs[1].legend(ncol=2,loc='upper left')

	# (c) PLOT S(t)
	# Plot mapped values from LVDT
	axs[2].plot(dtzindex,df_CM24['S tot'].values ,'b-',zorder=10)
	# Plot modeled values
	axs[2].plot(dtzindex,df_CM24['hat S tot'].values ,'r--',zorder=5)
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
	DAT = (df_NT24['Tau_kPa'],df_mu_obs['mu_obs'],\
		-df_Z24['LVDT_mm red'])

	for i_,D_ in enumerate(DAT):
	# 	# Pick extremum within given period
	# 	ex24 = pick_extrema_indices(D_,T=pd.Timedelta(24,unit='hour'))
	# 	# Get y-lims
		ylim = axs[i_].get_ylim()
	# 	# Plot minima times
	# 	for j_, t_ in enumerate(ex24['I_min']):
	# 		if i_ == 1 and j_ == 1:
	# 			pass
	# 		else:
	# 			axs[i_].plot(np.ones(2,)*(t_ - t0_T24).total_seconds()/3600,ylim,\
	# 						':',color='dodgerblue',zorder=1)
	# 	# Plot maxima times
	# 	for t_ in ex24['I_max']:
	# 		axs[i_].plot(np.ones(2,)*(t_ - t0_T24).total_seconds()/3600,ylim,\
	# 					 '-.',color='dodgerblue',zorder=1)
	# 	# re-enforce initial ylims
		# axs[i_].set_ylim(ylim)
	# 	# set xlims from data limits
		axs[i_].set_xlim((dtindex.min(),dtindex.max()))
		axs[i_].text(-5,(ylim[1] - ylim[0])*0.85 + ylim[0],\
					LBL[i_],fontsize=14,fontweight='extra bold',\
					fontstyle='italic',ha='right')

	axs[-1].set_xlabel('Elapsed Time from Start of Experiment T24 (hr)')

	axs[0].xaxis.set_ticklabels([])
	axs[1].xaxis.set_ticklabels([])
# if issave:
# 	OFILE = os.path.join(ODIR,'JGLAC_Fig05_Experiment_T24_Timeseries_%ddpi.%s'%(DPI,FMT.lower()))
# 	plt.savefig(OFILE,dpi=DPI,format=FMT.lower())

# plt.show()

	if not args.render_only:
		if args.dpi == 'figure':
			dpi = 'figure'
		else:
			try:
				dpi = int(args.dpi)

			except:
				dpi = 'figure'
		if dpi == 'figure':
			savename = os.path.join(args.output_path, f'JGLAC_Fig05_fdpi.{args.format}')
		else:
			savename = os.path.join(args.output_path, f'JGLAC_Fig05_{dpi}dpi.{args.format}')
		if not os.path.exists(os.path.split(savename)[0]):
			os.makedirs(os.path.split(savename)[0])
		plt.savefig(savename, dpi=dpi, format=args.format)

	if args.show:
		plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		prog='JGLAC_Fig05.py',
		description='Observed and modeled slip parameter time series from experiment T24'
	)


	parser.add_argument(
		'-i',
		'--input_path',
		dest='input_path',
		default=os.path.join('..','..','processed_data'),
		help='Path to processed data (several sub-sources)',
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