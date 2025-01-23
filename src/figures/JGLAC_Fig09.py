"""
:module: JGLAC_Fig09.py
:version: 1 - Revision for JOG-2024-0083 (Journal of Glaciology)
:short ref: Stevens and others - Experimental constraints on transient glacier slip with ice-bed separation
:Figure #: 9
:auth: Nathan T. Stevens
:email: ntstevenuw.edu
:license: CC-BY-4.0
:purpose: Plot change in drag againt effective pressure normalized slip velocity (sliding rule parameter space)
"""

import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(args):
	# Map Data
	# ROOT = os.path.join('..')
	# DDIR = os.path.join(ROOT,'processed_data','cavities')
	# Map Experimental Data
	T24_CM = os.path.join(args.gpath,'EX_T24_cavity_metrics.csv')
	T06_CM = os.path.join(args.gpath,'EX_T06_cavity_metrics.csv')
	MOD_NkPa = os.path.join(args.mpath,'SigmaN_kPa_grid.csv')
	MOD_TkPa = os.path.join(args.mpath,'Tau_kPa_grid.csv')
	DT_FILE = os.path.join(args.epath,'Delta_Tau_Estimates.csv')



	### LOAD EXPERIMENTAL DATA ###
	# Load Experimental Data
	df_T24 = pd.read_csv(T24_CM)
	df_T24.index = pd.to_datetime(df_T24.Epoch_UTC, unit='s')
	df_T06 = pd.read_csv(T06_CM)
	df_T06.index = pd.to_datetime(df_T06.Epoch_UTC, unit='s')
	df_Dt = pd.read_csv(DT_FILE)
	D_tau = df_Dt['Delta tau kPa'].mean()

	### LOAD STEADY STATE MODEL DATA ###
	df_Nmod = pd.read_csv(MOD_NkPa, index_col=[0])
	df_Tmod = pd.read_csv(MOD_TkPa, index_col=[0])
	NkPa_MOD = df_Nmod['15.000000'].values
	TkPa_MOD = df_Tmod['15.000000'].values

	### SET REFERENCES AND OFFSETS ###
	QSS_trim_24=4
	t0_T24 = pd.Timestamp('2021-10-26T18:58') + pd.Timedelta(24*QSS_trim_24, unit='hour')
	t1_T24 = t0_T24 + pd.Timedelta(24*(5- QSS_trim_24),unit='hour')
	df_T24 = df_T24[(df_T24.index >= t0_T24) & (df_T24.index < t1_T24)]
	QSS_trim_06 = 2
	t0_T06 = pd.Timestamp('2021-11-1T11:09:15') + pd.Timedelta(6*QSS_trim_06, unit='hour')
	t1_T06 = t0_T06 + pd.Timedelta(6*(5-QSS_trim_06), unit='hour')
	df_T06 = df_T06[(df_T06.index >= t0_T06) & (df_T06.index < t1_T06)]

	### CORRECT SHEAR STRESS
	df_tobs_T24 = df_T24['T kPa'] - D_tau
	df_tobs_T06 = df_T06['T kPa'] - D_tau
	### CALCULATE DRAG
	df_mu_T24 = df_tobs_T24/df_T24['N kPa']
	df_mu_T06 = df_tobs_T06/df_T06['N kPa']

	### STATE SLIP VELOCITY
	U_b = 15.

	### PLOTTING SECTION ###
	fig = plt.figure(figsize=(5,6))
	GS = fig.add_gridspec(ncols=1,nrows=2,hspace=0,wspace=0)
	axs = [fig.add_subplot(GS[_i]) for _i in range(2)]

	## MU
	axs[0].plot(np.r_[U_b/NkPa_MOD, np.zeros(1)],
		     np.r_[TkPa_MOD/NkPa_MOD, np.zeros(1)],
		  	'r-',label='Steady State Model')
	axs[0].plot(U_b/(df_T24['N kPa'].values),
		  	df_mu_T24.values,
			'k-',label=f'Exp. T24 Cycles {QSS_trim_24}$\\endash$5')
	axs[0].plot(15/(df_T06['N kPa'].values),
		  	df_mu_T06.values,
			'b-',label=f'Exp. T06 Cycles {QSS_trim_06}$\\endash$5')
	
	# axs[0].ylim(ylims)
	axs[0].legend(loc='lower right')
	axs[0].set_ylabel('Drag [$\mu$] ( - )')
	# axs[0].set_xlabel('$U_b / N$ ($m$ $kPa^{-1}$ $a^{-1}$)')
	axs[0].set_xticklabels([])

	## TAU
	axs[1].plot(np.r_[U_b/NkPa_MOD, np.zeros(1)],
		     np.r_[TkPa_MOD, np.zeros(1)],
		  	'r-',label='Steady State Model')
	axs[1].plot(U_b/(df_T24['N kPa'].values),
		  	df_tobs_T24.values,
			'k-',label=f'Exp. T24 Cycles {QSS_trim_24}$\\endash$5')
	axs[1].plot(15/(df_T06['N kPa'].values),
		  	df_tobs_T06.values,
			'b-',label=f'Exp. T06 Cycles {QSS_trim_06}$\\endash$5')
	

	
	## FORMATTING

	axs[0].legend(loc='lower right')
	axs[0].set_ylabel('Drag [$\mu$] ( - )')
	axs[1].set_ylabel('Shear Stress [$\\tau$] (kPa)')
	axs[1].set_xlabel('$U_b/ N$ (m kPa$^{-1}$ a$^{-1}$)\n[$U_b$ = 15 $m$ $a^{-1}$]')
	for _e, _c in enumerate(['a','b']):
		axs[_e].set_xlim([0., 0.1])
		axs[_e].text(axs[_e].get_xlim()[1]*0.01, axs[_e].get_ylim()[1]*0.92,
			   _c,fontweight='extra bold', fontstyle='italic', fontsize=14)
		axs[_e].grid(linestyle=':')


	if not args.render_only:
		if args.dpi == 'figure':
			dpi = 'figure'
		else:
			try:
				dpi = int(args.dpi)

			except:
				dpi = 'figure'
		if dpi == 'figure':
			savename = os.path.join(args.output_path, f'JGLAC_Fig09_fdpi.{args.format}')
		else:
			savename = os.path.join(args.output_path, f'JGLAC_Fig09_{dpi}dpi.{args.format}')
		if not os.path.exists(os.path.split(savename)[0]):
			os.makedirs(os.path.split(savename)[0])
		plt.savefig(savename, dpi=dpi, format=args.format)

	if args.show:
		plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		prog='JGLAC_Fig09.py',
		description='Cross plots of adjusted drag and slip/effective pressure ratios for experiments T24 and T06 and modeled values'
	)
	parser.add_argument(
		'-e',
		'--experiment_path',
		dest='epath',
		default=os.path.join('..','processed_data','experiments'),
		help='Path to processed data for split out experiment data',
		type=str
	)
	parser.add_argument(
		'-g',
		'--geometry_path',
		dest='gpath',
		default=os.path.join('..','processed_data','geometry'),
		help='Path to processed data for cavity geometries',
		type=str
	)
	parser.add_argument(
		'-m',
		'--model_path',
		dest='mpath',
		default=os.path.join('..','processed_data','model'),
		help='Path to modeled values',
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