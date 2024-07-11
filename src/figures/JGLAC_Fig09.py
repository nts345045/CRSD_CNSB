"""
:module: JGLAC_Fig09.py
:purpose: Plot the key timeseries from experiment T6 (6 hour oscillation)
:version: 0 - Submission format to Journal of Glaciology
:short ref: Stevens and others (submitted)
			Experimental constraints on transient glacier sliding with ice-bed separation
:Figure #: 9
:Figure Caption: 
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
"""

import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.model.lliboutry_kamb_model as lkm

## Plotting Subroutines ##
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

def get_lims(XI, YI, PADXY): 
	xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
			np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
	ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
			np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))
	return xlims, ylims

def main(args):
	# Map Data
	# ROOT = os.path.join('..')
	# DDIR = os.path.join(ROOT,'processed_data','cavities')
	# Map Experimental Data
	T24_CM = os.path.join(args.input_path,'cavities','EX_T24_cavity_metrics.csv')
	T06_CM = os.path.join(args.input_path,'cavities','EX_T06_cavity_metrics.csv')
	MOD_NkPa = os.path.join(args.input_path,'steady_state','SigmaN_kPa_grid.csv')
	MOD_TkPa = os.path.join(args.input_path,'steady_state','Tau_kPa_grid.csv')

	# MOD_CM = os.path.join(args.input_path,'modeled_values.csv')

	# Map output directory
	# ODIR = os.path.join(ROOT,'results','figures')

	### LOAD EXPERIMENTAL DATA ###
	# Load Experimental Data
	df_T24 = pd.read_csv(T24_CM)
	df_T24.index = pd.to_datetime(df_T24.Epoch_UTC, unit='s')
	df_T06 = pd.read_csv(T06_CM)
	df_T06.index = pd.to_datetime(df_T06.Epoch_UTC, unit='s')

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

	D_mu = 0.23
	xlims = [0, .1]
	ylims = [-D_mu, .12]
	D_tau = np.mean([84.38, 84.28]) #[kPa] - amount to reduce \tau


	### PLOTTING SECTION ###
	fig = plt.figure(figsize=(6,5.25))
	# GS = fig.add_gridspec(ncols=1,nrows=2,hspace=.2,wspace=0)
	# axs = [fig.add_subplot(GS[_i]) for _i in range(2)]
	plt.plot(np.r_[15/NkPa_MOD, np.zeros(1)],
		     np.r_[TkPa_MOD/NkPa_MOD, np.zeros(1)] - D_mu,
		  	'r-',label='Steady State Model')
	plt.plot(15/(df_T24['N kPa'].values),
		  	(df_T24['T kPa'].values - D_tau)/df_T24['N kPa'].values - D_mu,
			'k-',label=f'Exp. T24 Cycles {QSS_trim_24}$\\endash$5')
	plt.plot(15/(df_T06['N kPa'].values),
		  	(df_T06['T kPa'].values - D_tau)/df_T06['N kPa'].values - D_mu,
			'b-',label=f'Exp. T06 Cycles {QSS_trim_06}$\\endash$5')
	
	plt.xlim(xlims)
	plt.ylim(ylims)
	plt.legend(loc='lower right')
	plt.ylabel('Change in Drag ( - )')
	plt.xlabel('$U_b / N$ ($m$ $kPa^{-1}$ $a^{-1}$)')

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