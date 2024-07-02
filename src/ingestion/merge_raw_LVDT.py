"""
:module: merge_raw_LVDT.py
:auth: Nathan T. Stevens, Dougal D. Hansen
:email: ntsteven@uw.edu, ddhansen3@wisc.edu
:license: CC-BY-4.0
:purpose: 
	Merge individual LVDT (Linear Vector Displacement Transducer) 
	raw data files into one CSV and detect and and correct large 
	jumps in measurements arising from re-positioning the LVDT 
	during the experiment.
"""

import argparse, logging
import pandas as pd
import numpy as np

Logger = logging.getLogger('merge_raw_LVDT.py')


def read_lvdt_record(file):
	dt = (pd.Timestamp("1904-1-1") - pd.Timestamp("1970-1-1")).total_seconds()
	df_in = pd.read_csv(file, sep='\s+')
	measure = df_in['LVDT1.1'].values
	time = df_in['time'].values
	df_out = pd.DataFrame(
		{'LVDT_mm': measure,
		'epoch': time + dt},
		index = (time + dt).astype('datetime64[s]')
	)
	df_out.index.name = 'Time_UTC'
	return df_out

def main(input_files, threshold=0.05, range=1):
	"""Primary process

	:param input_file_string: glob-compliant string that will produce a list of input LVDT raw files, defaults to '*.txt'
	:type input_file_string: str, optional
	:param threshold: millimeter scale step size threshold for applying incremental offset corrections, defaults to 0.05
	:type threshold: float, optional
	:param range: number of offset samples to use for calculating step sizes, defaults to 1
	:type range: int, optional
	:return: output dataframe
	:rtype: pandas.core.dataframe.DataFrame
	"""	
	# Load and merge
	flist = input_files
	for _n, _f in enumerate(flist):
		Logger.info(f'reading: {_f}')
		if _n == 0:
			df = read_lvdt_record(_f)

		else:
			df = pd.concat(
				[df, read_lvdt_record(_f)],
				ignore_index=False,
				axis=0)
	# Sort result by timestamps
	df = df.sort_index()
	# Calculate first difference
	zv = df.LVDT_mm.copy().values
	dz = zv[range:] - zv[:-range]

	# Apply progressive shifts to trailing LVDT measurements
	for i_,dz_ in enumerate(dz):
		if np.abs(dz_) > threshold:
			zv[i_:] -= dz_
	
	df_out = pd.DataFrame(
		{'LVDT_mm': zv, 'epoch': df.epoch.values},
		index=df.index)
	# Do manually ID'd trim out of flat spot in data
	IND = (df_out.index >= pd.Timestamp("2021-11-06T19:30")) & (df_out.index <= pd.Timestamp("2021-11-07T18"))
	df_out = df_out[~IND]		
	return df_out


if __name__ == "__main__":
	# Set up logging to terminal
	ch = logging.StreamHandler()                                                            
	fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(fmt)
	Logger.addHandler(ch)
	Logger.setLevel(logging.INFO)

	# Setup command line argument parsing
	parser = argparse.ArgumentParser(
		prog='merge_raw_LVDT.py',
		description='concatenates timestamped LVDT observations and corrects for large jumps in measurements'
	)
	parser.add_argument(
		'-i',
		'--input',
		action='store',
		dest='input_files',
		nargs='+',
		default='./*.txt',
		help='List of file names to process',
		type=str
	)
	parser.add_argument(
		'-o',
		'--output',
		action='store',
		dest='output_file',
		default='./merged_destepped_LVDT__UTC_Timing.csv',
		help='output file to save results to'
	)
	parser.add_argument(
		'-t',
		'--thresh',
		action='store',
		dest='threshold',
		help='minimum millimeter-scaled stepwise change in LVDT measurements to be considered a step to correct',
		default=0.05,
		type=float
	)

	parser.add_argument(
		'-r',
		'--range',
		action='store',
		dest='rng',
		help='sampling range for calculating step size',
		default=1,
		type=int
	)

	arguments = parser.parse_args()
	df_out = main(
		input_files = arguments.input_files,
		threshold = arguments.threshold,
		range=arguments.rng)
	S_out = df_out['LVDT_mm']
	S_out.index = df_out['epoch']
	S_out.index.name='Epoch_UTC'
	S_out = S_out.sort_index()
	Logger.info(f'writing data to disk {arguments.output_file}')
	S_out.to_csv(arguments.output_file, header=True, index=True)
	Logger.info('data written to disk - concluding main')
			   
# ROOT = os.path.join('..','..','..')
# IROOT = os.path.join(ROOT,'raw','LVDT')
# OROOT = os.path.join(ROOT,'processed','timeseries')

