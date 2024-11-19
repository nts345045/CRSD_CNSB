"""
:module: JGLAC_Fig06.py
:version: 1 - Revision for JOG-2024-0083 Journal of Glaciology
:short ref: Stevens and others - Experimental constraints on transient glacier slip with ice-bed separation
:Figure #: 6
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: Plot vertical stresses, shear stress, 
	drag, and contact fractions time-series from experiment T06
	REV1 updates
	 - Add P_V and N as subplot (a)
	 - Add 48 hour rolling average values
	 - Add vertical and horizontal grids
"""
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(args):
	# Map Experimental Data Files
	T06_NT = os.path.join(args.epath,'EX_T06-Pressure.csv')
	T06_LV = os.path.join(args.epath,'EX_T06-LVDT-reduced.csv')
	T06_CM = os.path.join(args.gpath,'EX_T06_cavity_metrics.csv')
	T06_CP = os.path.join(args.gpath,'Postprocessed_Cavity_Geometries.csv')
	DT_file = os.path.join(args.epath, 'Delta_Tau_Estimates.csv')

	# LOAD EXPERIMENTAL DATA #
	df_NT06 = pd.read_csv(T06_NT)
	df_Z06 = pd.read_csv(T06_LV)
	df_CM06 = pd.read_csv(T06_CM)
	df_CP06 = pd.read_csv(T06_CP)
	df_Dt = pd.read_csv(DT_file, index_col=[0])


	df_NT06.index = pd.to_datetime(df_NT06.Epoch_UTC,unit='s')
	df_Z06.index = pd.to_datetime(df_Z06.Epoch_UTC, unit='s')
	df_CM06.index = pd.to_datetime(df_CM06.Epoch_UTC, unit='s')
	df_CP06.index = pd.to_datetime(df_CP06.Epoch_UTC, unit='s')


	# Define Reference time for T06
	t0_T06 = pd.Timestamp('2021-11-1T16:09:15')

	# Get average D_tau
	D_tauP = df_Dt['Delta tau kPa'].mean()
	
	# Make elapsed time indices
	# For stress measures
	dtindex = (df_NT06.index - t0_T06).total_seconds()/3600
	# For LVDT measures
	# dtzindex = (df_Z06.index - t0_T06).total_seconds()/3600
	dtzindex = (df_CM06.index - t0_T06).total_seconds()/3600

	### CALCULATE OBSERVED DRAG ###
	df_mu_obs = (df_NT06['Tau_kPa'] - D_tauP)/df_NT06['Pe_kPa']

	### CALCULATE STEADY-STATE MODEL DRAG ###
	df_mu_calc = df_CM06['hat T kPa']/df_CM06['N kPa']

	### PLOTTING SECTION ###
	# Initialize figure and subplot axes
	fig,axs = plt.subplots(ncols=1,nrows=4,figsize=(7.5,6))

	# (a) PLOT Pe
	axs[0].plot(dtindex, df_NT06['Pe_kPa'], 'k-', label='$N$', zorder=8)
	axs[0].plot(dtindex, df_NT06['SigmaN_kPa'], 'r-', label='$P_V$', zorder=5)

	axs[0].set_ylim([190, 540])
	# Label cycles
	for _c in range(5):
		axs[0].text(3+6*_c, 500, f'{_c+1}', ha='center')
	# axs[0].legend(ncols=2, loc='lower center', bbox_to_anchor=(0.585, -0.05)).set_zorder(level=1)
	# Plot long-term average
	axs[0].plot(
		dtindex, 
		df_NT06['Pe_kPa'].rolling(pd.Timedelta(12,unit='hour')).mean(),
		'k:')

	# (b) PLOT \tau(t) 
	# Plot observed values
	axs[1].plot(dtindex,df_NT06['Tau_kPa'] - D_tauP,'k-',zorder=8, label='$\\tau^{obs}$')
	# Plot modeled values
	axs[1].plot(dtzindex,df_CM06['hat T kPa'].values ,'r-',zorder=5, label='$\\tau^{calc}$')

	# Apply labels & formatting
	axs[1].set_ylabel('Shear Stress (kPa)')
	# Set custom y-limits and ticks
	ylims = axs[1].get_ylim()
	axs[1].set_ylim([ylims[0]-10, ylims[1]+10])
	axs[1].set_yticks([50, 100, 150])
	# axs[1].legend(ncols=2, loc='lower center', bbox_to_anchor=(0.585,-0.05)).set_zorder(level=1)
	
	axs[1].plot(
		dtindex, 
		(df_NT06['Tau_kPa'] - D_tauP).rolling(pd.Timedelta(12,unit='hour')).mean(),
		'k:'
	)
	axs[1].plot(
		dtzindex,
		df_CM06['hat T kPa'].rolling(pd.Timedelta(12, unit='hour')).mean(),
		'r:'
	)

	# (c) PLOT \mu
	# Plot observed values
	axs[2].plot(dtindex,df_mu_obs,'k-',zorder=5,label='$\\mu^{obs}$')
	# plot modeled values
	axs[2].plot(dtzindex,df_mu_calc,'r-',zorder=5,label='$\\mu^{calc}$')
	# Set custom y limits & ticks
	ylims = axs[2].get_ylim()
	axs[2].set_yticks([0.1, 0.15, 0.2, 0.25, 0.3])
	axs[2].set_ylim([ylims[0]-0.01, ylims[1]+0.02])
	# axs[2].legend(ncol=2,loc='lower right').set_zorder(level=1)

	axs[2].plot(
		dtindex, 
		df_mu_obs.rolling(pd.Timedelta(12, unit='hour')).mean(),
		'k:'
	)

	axs[2].plot(
		dtzindex,
		df_mu_calc.rolling(pd.Timedelta(12,unit='hour')).mean(),
		'r:'
	)

	# (d) PLOT S(t)
	# Plot mapped values from LVDT
	axs[3].plot(dtzindex,df_CM06['S tot'].values ,'k-',zorder=10, label='$S^{LVDT}$')
	# Plot modeled values
	axs[3].plot(dtzindex,df_CM06['hat S tot'].values ,'r-',zorder=5, label='$S^{calc}$')
	# Set custom y-limits
	axs[3].set_ylim([0.08, 0.32])
	axs[3].set_yticks(np.arange(0.1,0.3,0.05))
	# ylims = axs[3].get_ylim()
	# axs[3].set_ylim([ylims[0]-0.025, ylims[1]+0.025])
	# axs[3].legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, -0.05)).set_zorder(level=1)

	axs[3].plot(
		dtzindex,
		df_CM06['S tot'].rolling(pd.Timedelta(12, unit='hours')).mean(),
		'k:'
	)
	axs[3].plot(
		dtzindex,
		df_CM06['hat S tot'].rolling(pd.Timedelta(12, unit='hours')).mean(),
		'r:'
	)

	# ## SUBPLOT FORMATTING
	plt.subplots_adjust(hspace=0)
	LBL = ['a','b','c','d']
	YHN = ['Vertical Stress\n(kPa)','Shear Stress\n(kPa)','Drag ( - )','Contact Fraction\n( - )']
	for _e in range(len(axs)):
		ylim = axs[_e].get_ylim()
		# set xlims from data limits
		axs[_e].set_xlim((-1.5,dtindex.max()))
		# set xticks positions
		axs[_e].set_xticks(np.arange(0,33,3))
		axs[_e].set_xticks(np.arange(-1.5,33,1.5), minor=True)
		# Populate legends
		axs[_e].legend(ncols=2, loc='lower center',
				 bbox_to_anchor=(0.7325,-0.05)).set_zorder(level=1)

		# set yaxis label position
		axs[_e].yaxis.set_label_position('right')
		# Set ylabel
		axs[_e].set_ylabel(YHN[_e],rotation=270,labelpad=25)
		# set subplot label
		axs[_e].text(
			-0.5,
			(ylim[1] - ylim[0])*0.825 + ylim[0],
			LBL[_e],
			fontsize=14,
			fontweight='extra bold',
			fontstyle='italic',
			ha='right')
		# Turn on grids
		axs[_e].grid(True, which='both', linestyle=':')

	# Label xaxis for bottom plot & format major ticks
	axs[-1].set_xlabel('Elapsed Time During Exp. T06 (hr)')
	axs[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:d}'))
	# Remove all ticks for other plots
	axs[0].xaxis.set_ticklabels([])
	axs[1].xaxis.set_ticklabels([])
	axs[2].xaxis.set_ticklabels([])
	axs[0].set_title('Cycle Number',fontsize=10)

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
		'-e',
		'--experiment_path',
		dest='epath',
		default=os.path.join('..','..','processed_data','experiments'),
		help='Path to processed data for distinct experiments',
		type=str
	)
	
	parser.add_argument(
		'-g',
		'--geometry_path',
		dest='gpath',
		default=os.path.join('..','..','processed_data','geometry'),
		help='Path to processed data for cavitiy geometries',
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

