"""
:module: JGLAC_Fig03.py
:version: 0 - Submission format to Journal of Glaciology
:short ref: Stevens and others - Experimental constraints on transient glacier slip with ice-bed separation
:Figure #: 3
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: 
	Plot the effecitve pressure time-series for experiments T24, T06 and bounding hold periods
"""
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main(args):

	# df = pd.read_csv(os.path.join('..','processed_data','4_Smoothed_Pressure_Data.csv'))
	df = pd.read_csv(os.path.join(args.input_pressure_file))
	S_N = df['Pe_kPa']
	S_N.index = pd.to_datetime(df['Epoch_UTC'], unit='s')

	# Figure 3 rendering
	DPI = 200
	FMT = 'png'
	# Labels and intervals
	LBV = ['Hold Period\n(Mechanically Inferred)\n(Steady State)','T24','Hold Period\n(Geometric Steady State)','T06']#,'T96']
	TSV = [pd.Timestamp("2021-10-25T18:00:00"),pd.Timestamp('2021-10-26T18:56'),\
		pd.Timestamp("2021-10-31T19:03"),pd.Timestamp('2021-11-1T16:09:15')]
	TEV = [pd.Timestamp("2021-10-26T18:56"),pd.Timestamp("2021-10-31T19:03"),\
		pd.Timestamp('2021-11-1T16:09:15'),pd.Timestamp("2021-11-02T22:26:30")]

	xlims = [pd.Timestamp('2021-10-25'),TEV[-1] + pd.Timedelta(3,unit='hour')]

	fig = plt.figure(figsize=(7.5, 5.3))
	ax1 = fig.add_subplot(111)
	ax1.plot(S_N, 'k-')
	ylims = ax1.get_ylim()

	# Insert experiment labels
	for i_,l_ in enumerate(LBV):
		# Vertical lines
		ax1.plot([TSV[i_],TSV[i_]],ylims,'k:',alpha=0.5)
		ax1.plot([TEV[i_],TEV[i_]],ylims,'k:',alpha=0.5)
		# Labels
		if i_ in [0,2]:
			ax1.text(TSV[i_] + (TEV[i_] - TSV[i_])/2,275,LBV[i_],ha='center',va='center',rotation=90)
		else:
			ax1.text(TSV[i_] + (TEV[i_] - TSV[i_])/2,180,'Exp. {}'.format(LBV[i_]),ha='center', va='bottom')
	# Insert cycle numbers
	for i_ in range(5):
		ax1.text(TSV[1] + pd.Timedelta(6,unit='hour') + i_*pd.Timedelta(24,unit='hour'),450,i_+1,ha='center')
		ax1.text(TSV[3] + pd.Timedelta(3.1,unit='hour') + i_*pd.Timedelta(6,unit='hour'),517,i_ + 1, ha='center')


	ax1.set_xlim(xlims)
	ax1.set_ylim([170,530])
	ax1.xaxis.set_major_locator(mdates.DayLocator())
	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
	ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=np.arange(0,25,3,dtype=int)))
	ax1.format_xdata = mdates.DateFormatter('%d')
	ax1.set_xlabel('October 2021                           Day of Month                           November 2021')
	ax1.set_ylabel('Effective Pressure [N] (kPa)')

	if not args.render_only:
		if args.dpi == 'figure':
			dpi = 'figure'
		else:
			try:
				dpi = int(args.dpi)

			except:
				dpi = 'figure'
		if dpi == 'figure':
			savename = os.path.join(args.output_path, f'JGLAC_Fig03_fdpi.{args.format}')
		else:
			savename = os.path.join(args.output_path, f'JGLAC_Fig03_{dpi}dpi.{args.format}')
		if not os.path.exists(os.path.split(savename)[0]):
			os.makedirs(os.path.split(savename)[0])
		plt.savefig(savename, dpi=dpi, format=args.format)

	if args.show:
		plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		prog='JGLAC_Fig03.py',
		description='Observed effective pressure time series for Experiments T24 and T06'
	)


	parser.add_argument(
		'-i',
		'--input_pressure_file',
		dest='input_pressure_file',
		default=os.path.join('..','..','processed_data','4_Smoothed_Pressure_data.csv'),
		help='Path to the cleaned up effective pressure data file',
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