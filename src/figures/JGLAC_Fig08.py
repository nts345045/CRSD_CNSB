"""
:module: JGLAC_Fig08.py
:version: 1 - Revision for JOG-2024-0083 (Journal of Glaciology)
:short ref: Stevens and others - Experimental constraints on transient glacier slip with ice-bed separation
:Figure #: 8 - new addition
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: Plot the phase spaces for \\tau-N-S
			for experiments T24 and T06, 
			and modeled equivalents
"""

import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.util import plot_cycles, get_lims

def main(args):
	# Map Experimental Data
	T24_CM = os.path.join(args.gpath,'EX_T24_cavity_metrics.csv')
	T06_CM = os.path.join(args.gpath,'EX_T06_cavity_metrics.csv')
	MOD_CM = os.path.join(args.mpath,'cavity_geometry_mapping_values.csv')
	DT_FILE = os.path.join(args.epath,'Delta_Tau_Estimates.csv')
	### LOAD EXPERIMENTAL DATA ###
	# Load Experimental Data
	df_T24 = pd.read_csv(T24_CM)
	df_T24.index = pd.to_datetime(df_T24.Epoch_UTC, unit='s')
	df_T06 = pd.read_csv(T06_CM)
	df_T06.index = pd.to_datetime(df_T06.Epoch_UTC, unit='s')
	df_MOD = pd.read_csv(MOD_CM)
	df_MOD.index = df_MOD['N_Pa']
	df_Dt = pd.read_csv(DT_FILE)
	### SET REFERENCES AND OFFSETS ###
	# Start times in UTC
	t0_T24 = pd.Timestamp('2021-10-26T18:58')
	t0_T06 = pd.Timestamp('2021-11-1T16:09:15')

	# Shear stress adjustment for sockets
	D_tau = df_Dt['Delta tau kPa'].mean()
	df_tobs_T24 = df_T24['T kPa'] - D_tau
	df_tobs_T06 = df_T06['T kPa'] - D_tau

	# Calculate drag
	df_mu_T24 = df_tobs_T24/df_T24['N kPa']
	df_mu_T06 = df_tobs_T06/df_T06['N kPa']
	df_mu_calc = df_MOD['T_Pa']/df_MOD['N_Pa']

	### SET PLOTTING PARAMETERS
	cmaps = ['Blues_r','Purples_r','RdPu_r','Oranges_r','Greys_r']

	PADXY = 0.10

	### PLOTTING SECTION ###
	# Initialize figure and subplot axes
	fig = plt.figure(figsize=(7.5,8))
	GS = fig.add_gridspec(ncols=2,nrows=2,hspace=0.2,wspace=0)
	axs = [fig.add_subplot(GS[_i]) for _i in range(4)]


	### SUBPLOT (A) T24 N x tau^obs
	# Get data and modeled value vectors
	XI = df_T24['N kPa'].values
	XM = df_MOD.index.values*1e-3
	YI = df_tobs_T24.values
	YM = df_MOD['T_Pa'].values*1e-3
	II = df_T24.index
	# Get plot limits from data
	xlims, ylims = get_lims(XI, YI, PADXY)
	# Plot data
	chs = plot_cycles(axs[0],XI,YI,II, t0_T24, cmaps, ncycles=5,
				   T=pd.Timedelta(24,unit='hour'), zorder=10)
	# Plot model
	axs[0].plot(XM, YM, 'r-', zorder=5)
	# Set plot limits
	axs[0].set_xlim(xlims)
	axs[0].set_ylim(ylims)


	### SUBPLOT (B) T06 N x tau^obs
	# Get data and modeled value vectors
	XI = df_T06['N kPa'].values
	XM = df_MOD.index.values*1e-3
	YI = df_tobs_T06.values
	YM = df_MOD['T_Pa'].values*1e-3
	II = df_T06.index

	chs = plot_cycles(axs[1],XI,YI,II, t0_T06, cmaps, ncycles=5,
				   T=pd.Timedelta(6,unit='hour'), zorder=10)
	axs[1].plot(XM, YM, 'r-', zorder=5)
	# Set plot limits using limits from subplot (C)
	axs[1].set_xlim(xlims)
	axs[1].set_ylim(ylims)
	axs[1].text(420,0.18, '- Rising $N$ ->', ha='center',rotation=5)
	axs[1].text(370,0.22, '<- Falling $N$ -', ha='center', rotation=5)

	### SUBPLOT (E) T24 tau^obs x S
	# Get data and modeled value vectors
	XI = df_T24['S tot'].values
	XM = df_MOD['Stot'].values
	YI = df_tobs_T24.values
	YM = df_MOD['T_Pa'].values*1e-3
	II = df_T24.index

	# Get plot limits from data
	xlims, ylims = get_lims(XI, YI, PADXY)
	# # Overwrite x-axis limits with custom value to show more of the modeled area
	# xlims = [0.124, 0.275]
	# Plot data
	chs = plot_cycles(axs[2],XI,YI,II, t0_T24, cmaps, ncycles=5,
				   T=pd.Timedelta(24,unit='hour'), zorder=10)
	# Plot modeled values
	axs[2].plot(XM, YM, 'r-', zorder=5)
	# Set plot limits
	axs[2].set_xlim(xlims)
	axs[2].set_ylim(ylims)

	# axs[4].text(.21, .04 + 0.24, '$N_{min}$',ha='center', va='center')
	# axs[4].text(.255, -0.04 + 0.24, '$N_{max}$',ha='center', va='center')

	### SUBPLOT (F) T06 S x \mu
	# Get data and modeled value vectors
	XI = df_T06['S tot'].values
	XM = df_MOD['Stot'].values
	YI = df_tobs_T06.values
	YM = df_MOD['T_Pa'].values*1e-3
	II = df_T06.index
	# Plot data
	chs = plot_cycles(axs[3],XI,YI,II, t0_T06, cmaps, ncycles=5,
				   T=pd.Timedelta(6,unit='hour'), zorder=10)
	# Plot modeled values
	axs[3].plot(XM, YM, 'r-', zorder=5)
	# Set plot limits using limits from subplot (E)
	axs[3].set_xlim(xlims)
	axs[3].set_ylim(ylims)

	# ### SUBPLOT (C) T24 tau^obs x \mu
	# # Set data and model vectors
	# XI = df_mu_T24.values
	# XM = df_mu_calc.values
	# YI = df_tobs_T24.values
	# YM = df_MOD['T_Pa'].values*1e-3
	# II = df_T24.index
	# # Get plot limits from data values
	# xlims, ylims = get_lims(XI, YI, PADXY)
	# # Plot data
	# chs = plot_cycles(axs[4],XI,YI,II, t0_T24, cmaps, ncycles=5,
	# 			   T=pd.Timedelta(24,unit='hour'), zorder=10)
	# # Plot modeled values
	# axs[4].plot(XM, YM, 'r-', zorder=5)
	# # Set axis limits
	# axs[4].set_xlim(xlims)
	# axs[4].set_ylim(ylims)


	# ### SUBPLOT (D) T06 tau^obs x \mu
	# # Assign Data & Modeled Values to Plot
	# XI = df_mu_T06.values
	# XM = df_mu_calc.values
	# YI = df_tobs_T06.values
	# YM = df_MOD['T_Pa'].values*1e-3
	# II = df_T06.index
	# # Plot data
	# chs = plot_cycles(axs[5],XI,YI,II, t0_T06, cmaps, ncycles=5,
	# 			   T=pd.Timedelta(6,unit='hour'), zorder=10)
	# # Plot modeled values
	# axs[5].plot(XM, YM, 'r-', zorder=5)
	# # Set plot limits using same limits from subplot (A)
	# axs[5].set_xlim(xlims)
	# axs[5].set_ylim(ylims)

	# # axs[5].arrow(220,-0.03, 500-220,-0.03, width=0.001, head_width=0.01, head_length=0.03)
	# axs[5].text(330,-0.05,'- Rising $N$ ->', rotation=-8)
	# axs[5].text(310,0.03, '<- Falling $N$ -', rotation=-10)




	# axs[5].text(0.19, -0.04 + 0.232, '- Rising $N$ ->', rotation=-65, ha='center',va='center')
	# axs[5].text(0.23, 0.00 + 0.232, '<- Falling $N$ -', rotation=-80, ha='center',va='center')
	# axs[5].text(0.19, 0.03 + 0.232, '$N_{min}$', ha='center',va='center')
	# axs[5].text(0.22, -0.06 + 0.232, '$N_{max}$', ha='center',va='center')


	### FORMATTING & LABELING
	# Add experiment titles to each column head
	axs[0].set_title('Exp. T24')
	axs[1].set_title('Exp. T06')

	for _e in range(4):

		if _e < 2:
			axs[_e].set_xlabel('Effective Pressure [$N$] (kPa)')
		elif 2 <= _e < 4:
			axs[_e].set_xlabel('Contact Fraction [$S^{LVDT}$] ( - )')
		else:
			axs[_e].set_xlabel('Drag [$\mu^{obs}$] ( - )')

		if _e%2 == 0:
			axs[_e].set_ylabel('Shear Stress [$\\tau^{obs}$] (kPa)')
			# if _e%2 == 1:
			# 	axs[_e].yaxis.set_label_position('right')
		axs[_e].grid(linestyle=':')

	# Shift right column y-axis labels and ticks to right
	for _i in [1,3]:
		axs[_i].yaxis.tick_right()
		# # Included for completeness, commented out because unused for now
		# axs[_i].yaxis.set_label_position('right')

	# Add subplot labels
	for _i, _l in enumerate(['a','b','c','d']):
		xlims = axs[_i].get_xlim()
		ylims = axs[_i].get_ylim()
		axs[_i].text(
			(xlims[1] - xlims[0])*0.025 + xlims[0],
			(ylims[1] - ylims[0])*0.90 + ylims[0],
			_l, fontweight='extra bold', fontstyle='italic', fontsize=14)

	### COLORBAR PLOTTING ###
	cbar_placement = 'bottom'
	# Create timing colorbar
	for k_ in range(5):
		if cbar_placement.lower() == 'bottom':
			cax = fig.add_axes([.15 + (.70/5)*k_,.035,.70/5,.015])
		elif cbar_placement.lower() == 'top':
			cax = fig.add_axes([.15 + (.70/5)*k_,1 - .09,.70/5,.015])
		chb = plt.colorbar(chs[k_],cax=cax,orientation='horizontal',ticks=[0.99])
		if k_ == 2:
			if cbar_placement.lower() == 'bottom':
				cax.text(0.5,-1.5,'Cycle No.',ha='center',va='center',fontsize=10)
			elif cbar_placement.lower() == 'top':
				cax.text(0.5,2.5,'Cycle No.',ha='center',va='center')
		chb.ax.set_xticklabels(str(k_+1))

	### SAVING DECISION TREE ###
	if not args.render_only:
		if args.dpi == 'figure':
			dpi = 'figure'
		else:
			try:
				dpi = int(args.dpi)

			except:
				dpi = 'figure'
		if dpi == 'figure':
			savename = os.path.join(args.output_path, f'JOG-2024-0083.Figure8.fdpi.{args.format}')
		else:
			savename = os.path.join(args.output_path, f'JOG-2024-0083.Figure8.{dpi}dpi.{args.format}')
		if not os.path.exists(os.path.split(savename)[0]):
			os.makedirs(os.path.split(savename)[0])
		plt.savefig(savename, dpi=dpi, format=args.format)

	### DISPLAY DECISION POINT ###
	if args.show:
		plt.show()


### MAKE FIGURE RENDERING COMMAND LINE FRIENDLY ###
if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		prog='JGLAC_Fig08.py',
		description='Cross plots of sliding rule parameters and contact area parameters for experiments T24 and T06'
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



