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

import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
	T24_CM = os.path.join(args.input_path,'EX_T24_cavity_metrics.csv')
	T06_CM = os.path.join(args.input_path,'EX_T06_cavity_metrics.csv')
	MOD_CM = os.path.join(args.input_path,'modeled_values.csv')

	# Map output directory
	# ODIR = os.path.join(ROOT,'results','figures')

	### LOAD EXPERIMENTAL DATA ###
	# Load Experimental Data
	df_T24 = pd.read_csv(T24_CM)
	df_T24.index = pd.to_datetime(df_T24.Epoch_UTC, unit='s')
	df_T06 = pd.read_csv(T06_CM)
	df_T06.index = pd.to_datetime(df_T06.Epoch_UTC, unit='s')
	df_MOD = pd.read_csv(MOD_CM)
	df_MOD.index = df_MOD['N_Pa']

	### SET REFERENCES AND OFFSETS ###
	t0_T24 = pd.Timestamp('2021-10-26T18:58')
	t0_T06 = pd.Timestamp('2021-11-1T11:09:15')
	D_tau = np.mean([91.76, 90.43]) #60.95 # [kPa] - amount to reduce \tau(t)

	# D_tau = 90.43 # [kPa] Steadystate misfit value

	### SET PLOTTING PARAMETERS
	cmaps = ['Blues_r','Purples_r','RdPu_r','Oranges_r','Greys_r']
	# issave = True
	# DPI = 200
	# FMT = 'PNG'
	URC = (0.9,0.85)
	ULC = (0.05,0.90)
	PADXY = 0.05



	### PLOTTING SECTION ###
	fig = plt.figure(figsize=(7.5,10))
	GS = fig.add_gridspec(ncols=2,nrows=3,hspace=0.2,wspace=0)
	axs = [fig.add_subplot(GS[_i]) for _i in range(6)]
	# axs = [fig.add_subplot(GS[0,0]),fig.add_subplot(GS[0,1]),\
	# 	fig.add_subplot(GS[1,0]),fig.add_subplot(GS[1,1])]


	# (A) T24 N x \mu'
	XI = df_T24['N kPa'].values
	XM = df_MOD.index.values*1e-3
	YI = (df_T24['T kPa'].values - D_tau) / XI
	YM = (df_MOD['T_Pa'].values*1e-3 - D_tau) / XM
	II = df_T24.index

	xlims, ylims = get_lims(XI, YI, PADXY)

	chs = plot_cycles(axs[0],XI,YI,II, t0_T24, cmaps, ncycles=5,
				   T=pd.Timedelta(24,unit='hour'), zorder=10)
	axs[0].plot(XM, YM, 'r--', zorder=5)
	axs[0].set_xlim(xlims)
	axs[0].set_ylim(ylims)


	# (B) T06 N x \mu'
	XI = df_T06['N kPa'].values
	XM = df_MOD.index.values*1e-3
	YI = (df_T06['T kPa'].values - D_tau) / XI
	YM = (df_MOD['T_Pa'].values*1e-3 - D_tau) / XM
	II = df_T06.index

	# xlims, ylims = get_lims(XI, YI, PADXY)

	chs = plot_cycles(axs[1],XI,YI,II, t0_T06, cmaps, ncycles=5,
				   T=pd.Timedelta(6,unit='hour'), zorder=10)
	axs[1].plot(XM, YM, 'r--', zorder=5)
	axs[1].set_xlim(xlims)
	axs[1].set_ylim(ylims)



	# (C) T24 N x S LVDT
	XI = df_T24['N kPa'].values
	XM = df_MOD.index.values*1e-3
	YI = df_T24['S tot'].values
	YM = df_MOD['Stot'].values
	II = df_T24.index

	xlims, ylims = get_lims(XI, YI, PADXY)

	chs = plot_cycles(axs[2],XI,YI,II, t0_T24, cmaps, ncycles=5,
				   T=pd.Timedelta(24,unit='hour'), zorder=10)
	axs[2].plot(XM, YM, 'r--', zorder=5)
	axs[2].set_xlim(xlims)
	axs[2].set_ylim(ylims)


	# (D) T06 N x \mu'
	XI = df_T06['N kPa'].values
	XM = df_MOD.index.values*1e-3
	YI = df_T06['S tot'].values
	YM = df_MOD['Stot'].values
	II = df_T06.index

	# xlims, ylims = get_lims(XI, YI, PADXY)

	chs = plot_cycles(axs[3],XI,YI,II, t0_T06, cmaps, ncycles=5,
				   T=pd.Timedelta(6,unit='hour'), zorder=10)
	axs[3].plot(XM, YM, 'r--', zorder=5)
	axs[3].set_xlim(xlims)
	axs[3].set_ylim(ylims)


	# (E) T24 N x S LVDT
	XI = df_T24['S tot'].values
	XM = df_MOD['Stot'].values
	YI = (df_T24['T kPa'].values - D_tau) / df_T24['N kPa'].values
	YM = (df_MOD['T_Pa'].values*1e-3 - D_tau) / (df_MOD['N_Pa'].values*1e-3)
	II = df_T24.index

	xlims, ylims = get_lims(XI, YI, PADXY)

	chs = plot_cycles(axs[4],XI,YI,II, t0_T24, cmaps, ncycles=5,
				   T=pd.Timedelta(24,unit='hour'), zorder=10)
	axs[4].plot(XM, YM, 'r--', zorder=5)
	axs[4].set_xlim(xlims)
	axs[4].set_ylim(ylims)


	# (F) T06 N x \mu'

	XI = df_T06['S tot'].values
	XM = df_MOD['Stot'].values
	YI = (df_T06['T kPa'].values - D_tau) / df_T06['N kPa'].values
	YM = (df_MOD['T_Pa'].values*1e-3 - D_tau) / (df_MOD['N_Pa'].values*1e-3)
	II = df_T06.index

	# xlims, ylims = get_lims(XI, YI, PADXY)

	chs = plot_cycles(axs[5],XI,YI,II, t0_T06, cmaps, ncycles=5,
				   T=pd.Timedelta(6,unit='hour'), zorder=10)
	axs[5].plot(XM, YM, 'r--', zorder=5)
	axs[5].set_xlim(xlims)
	axs[5].set_ylim(ylims)


	# Formatting & Labels
	axs[0].set_title('Experiment T24')
	axs[1].set_title('Experiment T06')

	for _i in range(4):
		axs[_i].set_xlabel('Effective Pressure [$N$] (kPa)')
	
	for _i in [4,5]:
		axs[_i].set_xlabel('Contact Fraction [$S_{LVDT}$] ( - )')
	

	for _i in [1,3,5]:
		axs[_i].yaxis.tick_right()
		# axs[_i].yaxis.set_label_position('right')
	for _i in [0,4]:
		axs[_i].set_ylabel('Drag [$\mu^\prime$ | $\\mu_{mod}$] ( - )')
	axs[2].set_ylabel('Contact Fraction [$S_{LVDT}$] ( - )')
	
	for _i, _l in enumerate(['a','b','c','d','e','f']):
		xlims = axs[_i].get_xlim()
		ylims = axs[_i].get_ylim()
		axs[_i].text(
			(xlims[1] - xlims[0])*0.025 + xlims[0],
			(ylims[1] - ylims[0])*0.90 + ylims[0],
			_l, fontweight='extra bold', fontstyle='italic', fontsize=14)

	### COLORBAR PLOTTING ###

	Tc = 24
	cbar_placement = 'bottom'
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


	if not args.render_only:
		if args.dpi == 'figure':
			dpi = 'figure'
		else:
			try:
				dpi = int(args.dpi)

			except:
				dpi = 'figure'
		if dpi == 'figure':
			savename = os.path.join(args.output_path, f'JGLAC_Fig07_fdpi.{args.format}')
		else:
			savename = os.path.join(args.output_path, f'JGLAC_Fig07_{dpi}dpi.{args.format}')
		if not os.path.exists(os.path.split(savename)[0]):
			os.makedirs(os.path.split(savename)[0])
		plt.savefig(savename, dpi=dpi, format=args.format)

	if args.show:
		plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		prog='JGLAC_Fig07.py',
		description='Cross plots of sliding rule parameters and contact area parameters for experiments T24 and T06'
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



