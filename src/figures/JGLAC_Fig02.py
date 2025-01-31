"""
:module: JGLAC_Fig02.py

:version: 0 - Submission format to Journal of Glaciology
:short ref: Stevens and others - Experimental constraints on transient glacier slip with ice-bed separation
:Figure #: 2
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: 
	Plot the modeled parameter space (N,\\tau,S,\\mu) for the UW-CRSD assuming rheologic properties identical to those in Zoet & Iverson (2015)
"""
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(args):
	path = args.input_path
	# path = os.path.join('..','results','model')
	df_N = pd.read_csv(os.path.join(path,'SigmaN_kPa_grid.csv'), index_col=[0])
	df_U = pd.read_csv(os.path.join(path,'Slip_mpy_grid.csv'), index_col=[0])
	df_S = pd.read_csv(os.path.join(path,'S_total_grid.csv'), index_col=[0])
	df_T = pd.read_csv(os.path.join(path,'Tau_kPa_grid.csv'), index_col=[0])
	# Calculate Drag
	df_u = df_T/df_N
	df_u[~np.isfinite(df_u.values)] = np.nan

	# Initialize Figure
	plt.figure(figsize=(6,4.5))

	# Plot drag background color
	drag_ch = plt.pcolor(df_U.values, df_N.values, df_u.values, cmap='Blues')
	
	# Render drag colorbar
	plt.colorbar(drag_ch)
	plt.text(38,500,'Drag [$\\mu$] ( - )',rotation=270,fontsize=10,ha='center',va='center')
	plt.clim([0,.35])

	# Shear stress contours
	tau_ch = plt.contour(df_U.values, df_N.values, df_T.values,
					levels=np.arange(25,275,25), colors=['k'])
	# Max shear stress contour
	max_tau_ch = plt.contour(df_U.values, df_N.values, df_T.values,
					levels=[275], colors=['r'], linestyles='--')

	contact_fract_ch = plt.contour(df_U.values, df_N.values, df_S.values,
					levels=np.arange(0.1,1,0.1), colors=['w'], linestyles=':')


	# Plot operational range for N(t)
	plt.plot([15]*2,[210,490],linewidth=4,color='orange',zorder=9,alpha=1)
	plt.plot(15,350,'d',color='orange',markersize=14,alpha=1)

	# Annotate about cavities
	## Technical Note: This isn't quite right because the bed is kambered, making for very small cavities on the outer radius of the bed.
	plt.text(0.25,800,'No Cavities\n($S\\approx$ 1)',fontsize=10,va='center')
	plt.text(15,50,'Large Cavities ($S$ < 0.1)', ha='center', va='center')
	

	# Plot V < V_{min} zone
	plt.fill_between([0,4],[0]*2,[900]*2,color='black',alpha=0.1)

	# Axis Labels
	plt.xlabel('Linear Sliding Velocity [$U_b$] (m a$^{-1}$)')
	plt.ylabel('Effective Pressure [$N$] (kPa)')
	plt.xlim([0, 30])

	# Plot shear stress contour labels
	mlocs = []; cxloc = 23

	# Custom contour label positions
	for level in tau_ch.collections:
		path = level.get_paths()
		if len(path) > 0:
			cxpath = path[0].vertices[:,0]
			cypath = path[0].vertices[:,1]
			cxidx = np.argmin(np.abs(cxpath - cxloc))
			cyloc = cypath[cxidx]
			mlocs.append((cxloc,cyloc))
			# mlocs.append((np.mean(level.get_paths()[0].vertices[:,0]),\
			# 	 		  np.mean(level.get_paths()[0].vertices[:,1])))
	plt.clabel(tau_ch,inline=True,inline_spacing=2,fontsize=10,fmt='%d kPa',manual=mlocs)

	# Format contour label for max tau
	mlocs = []
	for level in max_tau_ch.collections:
		path = level.get_paths()
		if len(path) > 0:
			cxpath = path[0].vertices[:,0]
			cypath = path[0].vertices[:,1]
			cxidx = np.argmin(np.abs(cxpath - cxloc))
			cyloc = cypath[cxidx]
			mlocs.append((cxloc,cyloc))
			# mlocs.append((np.mean(level.get_paths()[0].vertices[:,0]),\
			# 	 		  np.mean(level.get_paths()[0].vertices[:,1])))
	# Render contour label for max tau
	plt.clabel(max_tau_ch,inline=True,inline_spacing=2,fontsize=10,fmt='%d kPa',manual=mlocs)

	# Format contour labels for S
	mlocs = []; cxloc = 9
	for level in contact_fract_ch.collections:
		path = level.get_paths()
		if len(path) > 0:
			cxpath = path[0].vertices[:,0]
			cypath = path[0].vertices[:,1]
			cxidx = np.argmin(np.abs(cxpath - cxloc))
			cyloc = cypath[cxidx]
			mlocs.append((cxloc,cyloc))
			# mlocs.append((np.mean(path[0].vertices[:,0]),\
			# 	 		  np.mean(path[0].vertices[:,1])))
	# Render contour labels for S
	plt.clabel(contact_fract_ch,inline=True,inline_spacing=2,fontsize=10,fmt='%.1f',manual=mlocs)

	# Parse render_only argument ()
	if not args.render_only:
		# Handle if dpi is not specified or specified as 'figure'
		if args.dpi == 'figure':
			dpi = 'figure'
		else:
			try:
				dpi = int(args.dpi)

			except:
				dpi = 'figure'
		# Format output name
		if dpi == 'figure':
			savename = os.path.join(args.output_path, f'JOG-2024-0083.Figure2.fdpi.{args.format}')
		else:
			savename = os.path.join(args.output_path, f'JOG-2024-0083.Figure2.{dpi}dpi.{args.format}')

		# Check that save directory exists, if not make one
		if not os.path.exists(os.path.split(savename)[0]):
			os.makedirs(os.path.split(savename)[0])
		## SAVE FIGURE TO DISK ##
		plt.savefig(savename, dpi=dpi, format=args.format)

	# If show, render plot #
	if args.show:
		plt.show()


## RUN AS MAIN ##
if __name__ == '__main__':
	
	# Initialize parser
	parser = argparse.ArgumentParser(
		prog='JGLAC_Fig02.py',
		description='A steady state parameter space for the Lliboutry/Kamb sliding law for ice over an undulatory bed'
	)
	# Input file path argument
	parser.add_argument(
		'-i',
		'--input_path',
		dest='input_path',
		default=os.path.join('.','processed_data','steadystate'),
		help='Path to pandas *grid.csv files generated by src/primary/generate_parameter_space.py',
		type=str
	)
	# Output file path argument
	parser.add_argument(
		'-o',
		'--output_path',
		action='store',
		dest='output_path',
		default=os.path.join('..','results','figures'),
		help='path and name to save the rendered figure to, minus format (use -f for format). Defaults to "../results/figures/JGLAC_Fig01c"',
		type=str
	)
	# Output file format argument
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
	# Output file resolution argument
	parser.add_argument(
		'-d',
		'--dpi',
		action='store',
		dest='dpi',
		default='figure',
		help='set the `dpi` argument for :meth:`~matplotlib.pyplot.savefig. Defaults to "figure"'
	)
	# Render figure on desktop bool switch
	parser.add_argument(
		'-s',
		'--show',
		action='store_true',
		dest='show',
		help='if included, render the figure on the desktop in addition to saving to disk'
	)
	# Only render figure on desktop bool switch
	parser.add_argument(
		'-r',
		'--render_only',
		dest='render_only',
		action='store_true',
		help='including this flag skips saving to disk (i.e., do NOT save the figure to disk)'
	)
	# Parse arguments
	args = parser.parse_args()
	# Run main
	main(args)