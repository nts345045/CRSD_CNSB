"""
:script: processing/preprocess/smooth_timeseries.py
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: Apply one or more moving window average boxcar
    smoothing operators to 
"""

import argparse, logging
import pandas as pd
import numpy as np

Logger = logging.getLogger('smooth_timeseries.py')

def main():
    parser = argparse.ArgumentParser(
        prog='smooth_timeseries.py',
        description='use a moving window z-score metric to remove anomalous spikes in data'
    )
    parser.add_argument(
        '-i',
        '--input',
        action='store',
        dest='input_file',
        default=None,
        help='path and filename of file to smooth',
        type=str
    )
    parser.add_argument(
        '-o',
        '--output',
        action='store',
        dest='output_file',
        default='tmp_smoothed_data.csv',
        help='path and filename of file to save results to',
        type=str
    )
    parser.add_argument(
        '-n',
        '--niter',
        dest='niter',
        default=3,
        help='number of smoothing iterations to apply',
        type=float
    )
    parser.add_argument(
        '-w',
        '-window',
        dest='window',
        help='smoothing window length (seconds)',
        default=10.,
        type=float
    )
    args = parser.parse_args()
    Logger.info(f'loading data from: {args.input_file}')
    Logger.info(f'writing data to: {args.output_file}')
    Logger.info(f'using a window length of {args.window} sec')
    Logger.info(f'running {args.niter} iterations of window-centered boxcar smoothing')
    
    df = pd.read_csv(args.input_file, parse_dates=['Time_UTC'], index_col=[0])
    Logger.info('data loaded')
    for _n in range(args.niter):
        if _n == 0:
            df_out = df.copy().rolling(pd.Timedelta(args.window,unit='seconds')).mean()
        else:
            df_out = df_out.rolling(pd.Timedelta(args.window, unit='seconds')).mean()
        df_out.index -= pd.Timedelta(args.window, unit='seconds')/2.
        df_out= df_out[(df_out.index >= df.index.min())&
                    (df_out.index <= df.index.max())]
    
    df_out = pd.DataFrame()
    for _n, _c in enumerate(df.columns):
        if _c not in ['epoch','Time_UTC']:
            Logger.info(f'processing {_c}')
            s_ = df[_c].copy().rolling(dt).median()
            s_.index -= dt/2.
            s_Z = (df[_c] - s_)
            IND = np.abs(s_Z) <= args.threshold*s_Z.std()
            if _n == 0:
                df_out = df_out._append(df[_c][IND]).T
            else:
                df_out = pd.concat([df_out, df[_c][IND]], axis=1, ignore_index=False)
    df_out = df_out.assign(epoch=lambda x: (x.index - pd.Timestamp('1970-1-1')).total_seconds())
    Logger.info('writing data to disk')
    df_out.to_csv(args.output_file, header=True, index=True)
    Logger.info('data written to disk - concluding main')


if __name__ == '__main__':
    ch = logging.StreamHandler()                                                            
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    Logger.addHandler(ch)
    Logger.setLevel(logging.INFO)
    main()