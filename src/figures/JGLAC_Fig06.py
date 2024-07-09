"""
:module: JGLAC_Fig06_Experiment_T06_Timeseries.py
:purpose: Plot the key timeseries from experiment T06 (06 hour oscillation)
:version: 0 - Submission format to Journal of Glaciology
:short ref: Stevens, Hansen, and others
			Experimental constraints on transient glacier sliding with ice-bed separation
:Figure #: 6
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
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(args):
	ROOT = os.path.join('..')
	DDIR = os.path.join(ROOT,'processed_data')
	# Map Experimental Data Files
	T06_NT = os.path.join(args.input_path,'5_split_data','EX_T06-Pressure.csv')
	T06_LV = os.path.join(args.input_path,'6_lvdt_melt_corrected','EX_T06-LVDT-reduced.csv')
	T06_CM = os.path.join(args.input_path,'cavities','EX_T06_cavity_metrics.csv')
	T06_CP = os.path.join(args.input_path,'cavities','Postprocessed_Cavity_Geometries.csv')
	# Map output directory
	# ODIR = os.path.join(ROOT,'results','figures')

	# LOAD EXPERIMENTAL DATA #
	df_NT = pd.read_csv(T06_NT)
	df_Z = pd.read_csv(T06_LV)
	df_CM = pd.read_csv(T06_CM)
	df_CP = pd.read_csv(T06_CP)

	df_NT.index = pd.to_datetime(df_NT.Epoch_UTC,unit='s')
	df_Z.index = pd.to_datetime(df_Z.Epoch_UTC, unit='s')
	df_CM.index = pd.to_datetime(df_CM.Epoch_UTC, unit='s')
	df_CP.index = pd.to_datetime(df_CP.Epoch_UTC, unit='s')


	# Define Reference time for T06
	t0_T06 = pd.Timestamp('2021-11-1T16:09:15')
	D_tau = 0 #60.95 # [kPa] difference between observed \tau during steady state
				#       minus calculated values given the geometry of the bed
				# 	  for steady-state N(t)

	D_tauP = np.nanmean(df_CM[df_CM.index > t0_T06 + pd.Timedelta(30.01,unit='hour')]['T kPa']) - \
			np.nanmean(df_CM[df_CM.index > t0_T06 + pd.Timedelta(30.01,unit='hour')]['hat T kPa'])
	print('Estimated shift for \\tau is %.3e kPa'%(D_tauP))

	# Make elapsed time indices
	# For stress measures
	dtindex = (df_NT.index - t0_T06).total_seconds()/3600
	# For LVDT measures
	dtzindex = (df_Z.index - t0_T06).total_seconds()/3600
	dtmindex = (df_CM.index - t0_T06).total_seconds()/3600

	### CALCULATE OBSERVED DRAG ###
	# Observed Drag with \Delta \tau offset
	mu_obs = (df_NT['Tau_kPa'].values - D_tauP)/df_NT['Pe_kPa'].values
	df_mu_obs = pd.DataFrame({'mu_obs':mu_obs},index=df_NT.index)
	# Get Reference \\mu(t) value for differencing
	mu0_obs = df_mu_obs[dtindex <= 0].mean().values
	# Get observed change in drag
	Dmu_obs = mu_obs - mu0_obs

	### CALCULATE STEADY-STATE MODEL DRAG ###
	mu_calc = df_CM['hat T kPa'].values/df_CM['N kPa'].values
	# Get reference \\mu(t) for differencing
	mu0_calc = mu_calc[-1]
	# Get calculated change in drag
	Dmu_calc = mu_calc - mu0_calc

	### CALCULATE SHIFTED DRAG ###
	mu_tP = (df_NT['Tau_kPa'].values - D_tauP)/df_NT['Pe_kPa'].values



	### PLOTTING SECTION ###
	# Initialize figure and subplot axes
	fig,axs = plt.subplots(ncols=1,nrows=3,figsize=(7.5,5.5))


	# (a) PLOT \tau(t) 
	# Plot observed values
	axs[0].plot(dtindex,df_NT['Tau_kPa'],'k-',zorder=10,label='$\\tau^{obs}$')
	# Plot adjusted values
	axs[0].plot(dtindex,df_NT['Tau_kPa'] - D_tauP,'b-',zorder=7, label='$\\tau^{\\prime}$')
	# Plot modeled values
	axs[0].plot(dtmindex,df_CM['hat T kPa'].values,'r-',zorder=5, label='$\\tau^{calc}$')  
	# Apply labels & formatting
	axs[0].set_ylabel('Shear Stress (kPa)')
	# axb.set_ylabel('$\\hat{\\tau}$(t) [kPa]',rotation=270,labelpad=15,color='red')
	axs[0].set_xticks(np.arange(-3,36,3))
	axs[0].grid(axis='x',linestyle=':')

	axs[0].text(-0.25,df_CM['hat T kPa'].values[-1] + D_tauP/2,'$\\Delta \\tau$',fontsize=14,ha='center',va='center')
	axs[0].arrow(-1.33,162.33,0,95.4 - 162.33,head_width=2/5,width=0.1/5,head_length=10,fc='k')
	# Add legend
	ylims = axs[0].get_ylim()
	axs[0].set_ylim([ylims[0]-30, ylims[1]])
	axs[0].legend(ncols=3, loc='lower center', bbox_to_anchor=(0.5,-0.05)).set_zorder(level=1)
	# Annotation about logging gap
	axs[0].text(29,95,'Transducer Logging Gap',rotation=90, ha='center',va='center',fontsize=8)


	# (b) PLOT \Delta \mu(t)
	# Plot observed values
	axs[1].plot(dtindex,mu_tP,'b-',zorder=7,label='$\\mu^{\\prime}$')
	# plot modeled values
	axs[1].plot(dtmindex,mu_calc,'r-',zorder=5,label='$\\mu^{calc}$')
	# Apply labels & formatting
	axs[1].set_xticks(np.arange(-3,36,3))
	axs[1].grid(axis='x',linestyle=':')
	axs[1].set_ylabel('Drag ( - )')
	# Add legend
	ylims = axs[1].get_ylim()
	axs[1].set_ylim([ylims[0], ylims[1]+0.02])
	axs[1].legend(ncol=2,loc='upper right').set_zorder(level=1)


	# (c) PLOT S(t)
	# Plot mapped values from LVDT
	axs[2].plot(dtmindex,df_CM['S tot'],'b-',zorder=10, label='$S^{LVDT}$')
	# Plot modeled values
	axs[2].plot(dtmindex,df_CM['hat S tot'],'r-',zorder=5, label='$S^{calc}$')
	# Apply labels and formatting
	axs[2].set_xticks(np.arange(-3,36,3))
	axs[2].grid(axis='x',linestyle=':')
	axs[2].set_ylabel('Contact Fraction ( - )')
	# Add Legend
	ylims = axs[2].get_ylim()
	axs[2].set_ylim([ylims[0]-0.025, ylims[1]])
	axs[2].legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, -0.05)).set_zorder(level=1)


	# ## SUBPLOT FORMATTING
	plt.subplots_adjust(hspace=0)
	LBL = ['a','b','c']
	DAT = (df_NT['Tau_kPa'],df_mu_obs['mu_obs'],\
		-df_Z['LVDT_mm red'])

	ex_lims = [t0_T06 + pd.Timedelta(15,unit='minute'), t0_T06 + pd.Timedelta(28,unit='hour')]

	for i_,D_ in enumerate(DAT):
		axs[i_].set_xlim([-2.9,31])
		# Pick extremum within given period
		# ex6 = pick_extrema_indices(D_,T=pd.Timedelta(5.5,unit='hour'))
		# # Get y-lims
		ylim = axs[i_].get_ylim()
		# # Plot minima times
		# for t_ in ex6['I_min']:
		# 	if ex_lims[0] <= t_ <= ex_lims[1]:
		# 		axs[i_].plot(np.ones(2,)*(t_ - t0_T06).total_seconds()/3600,ylim,\
		# 					 ':',color='dodgerblue',zorder=1)
		# # Plot maxima times
		# for t_ in ex6['I_max']:
		# 	if ex_lims[0] <= t_ <= ex_lims[1]:
		# 		axs[i_].plot(np.ones(2,)*(t_ - t0_T06).total_seconds()/3600,ylim,\
		# 					 '-.',color='dodgerblue',zorder=1)
		# re-enforce initial ylims
		axs[i_].set_ylim(ylim)
		# set xlims from data limits
		# axs[i_].set_xlim((dtindex.min(),dtindex.max()))
		axs[i_].text(-2,(ylim[1] - ylim[0])*0.05 + ylim[0],\
					LBL[i_],fontsize=14,fontweight='extra bold',\
					fontstyle='italic')
		if LBL[i_] != 'c':
			axs[i_].xaxis.set_tick_params(labelbottom=False)



	axs[-1].set_xlabel('Elapsed Time from Start of Exp. T06 (hr)')

	if not args.render_only:
		if args.dpi == 'figure':
			dpi = 'figure'
		else:
			try:
				dpi = int(args.dpi)

			except:
				dpi = 'figure'
		if dpi == 'figure':
			savename = os.path.join(args.output_path, f'JGLAC_Fig06_fdpi.{args.format}')
		else:
			savename = os.path.join(args.output_path, f'JGLAC_Fig06_{dpi}dpi.{args.format}')
		if not os.path.exists(os.path.split(savename)[0]):
			os.makedirs(os.path.split(savename)[0])
		plt.savefig(savename, dpi=dpi, format=args.format)

	if args.show:
		plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		prog='JGLAC_Fig06.py',
		description='Observed and modeled slip parameter time series from experiment T06'
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

