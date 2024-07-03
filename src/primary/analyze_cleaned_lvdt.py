"""
:script: processing/primanry/analyze_cleaned_lvdt.py
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: 
"""

import argparse, logging, glob, os, sys
import pandas as pd
import numpy as np

Logger = logging.getLogger('smooth_timeseries.py')

def pick_extrema_indices(series,T=pd.Timedelta(6,unit='hour'),granularity=0.2):
	if isinstance(series,pd.DataFrame):
		series = series[series.columns[0]]

	t0 = series.index.min()
	tf = series.index.max()
	ti = t0
	min_ind = []; min_vals = []
	max_ind = []; max_vals = []
	while ti + T <= tf:
		iS = series[(series.index >= ti) & (series.index <=ti + T)]
		imax = iS.max(); Imax = iS.idxmax()
		imin = iS.min(); Imin = iS.idxmin()
		# If maximum value is greater than edge values
		if imax > iS.values[0] and imax > iS.values[-1]:
			# And maximum index is not already in the output list
			if Imax not in max_ind:
				max_ind.append(Imax)
				max_vals.append(imax)
		if imin < iS.values[0] and imin < iS.values[-1]:
			if Imin not in min_ind:
				min_ind.append(Imin)
				min_vals.append(imin)

		ti += T*granularity

	return {'I_max':pd.DatetimeIndex(max_ind),'V_max':np.array(max_vals),'I_min':pd.DatetimeIndex(min_ind),'V_min':np.array(min_vals)}


def fit_dateline(datetimes,values):
	# Convert datetime array into total seconds relative to minimum datetime
	tvsec = (datetimes - datetimes.min()).total_seconds()
	# Get slope
	mod = np.polyfit(tvsec,values,1)
	return mod[0]


def reduce_series_by_slope(series,slope,reference_time,reference_value):
	if isinstance(series,pd.DataFrame):
		series = series[series.columns[0]]
	series = series.sort_index()
	dt_ind = (series.index - reference_time).total_seconds()
	poly = [slope,reference_value]
	mfun = np.poly1d(poly)
	y_hat = mfun(dt_ind)
	y_red = series.values - y_hat
	s_out = pd.Series(y_red,index=series.index,name='%s red'%(series.name))
	return s_out



def main():
    parser = argparse.ArgumentParser(
        prog='smooth_timeseries.py',
        description='use a moving window z-score metric to remove anomalous spikes in data'
    )
    parser.add_argument(
        '-i',
        '--input_path',
        action='store',
        dest='input_path',
        default=None,
        help='path and filename of file to smooth',
        type=str
    )
    parser.add_argument(
        '-o',
        '--output_path',
        action='store',
        dest='output_path',
        default='tmp/processed_LVDT',
        help='path and filename of file to save results to',
        type=str
    )

    parser.add_argument(
        '-f',
        '--parameter_file',
        action='store',
        dest='params',
        default=None,
        help='parameter file for experiment-specific processing parameters',
        type=str
    )
    args = parser.parse_args()
    if args.params is None:
        Logger.critical('No parameter file provided with flag -f. Quitting')
        sys.exit(1)
    else:
        df_pf = pd.read_csv(args.params)
		
    # Get names of LVDT files
    flist = glob.glob(os.path.join(args.input_path,'*LVDT*.csv'))
    print(flist)
    breakpoint()
    for _f in flist:
        # Load experiment specific data
        df = pd.read_csv(_f)
        df.index = pd.to_datetime(df.Epoch_UTC)
		
        Logger.info(f'data loaded for {_f}')
        # Experiment specific parameter file
        IND = 
        peaks = pick_extrema_indices()


if __name__ == '__main__':
    ch = logging.StreamHandler()                                                            
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    Logger.addHandler(ch)
    Logger.setLevel(logging.INFO)
    main()