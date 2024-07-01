"""
:script: processing/preprocess/despike_timeseries.py
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: Synchronizes sampling rate of timeseries
    and interpolate into uniform temporal sampling
    with a sample spacing determined as the median
    inter-sample time for the timeseries data.
"""

import argparse, logging
import pandas as pd
import numpy as np

Logger = logging.getLogger('despike_timeseries.py')

def main():
    parser = argparse.ArgumentParser(
        prog='despike_timeseries.py',
        description='use a moving window z-score metric to remove anomalous spikes in data'
    )
    parser.add_argument(
        '-i',
        '--input',
        action='store',
        dest='input_file',
        default=None,
        help='path and filename of file to despike',
        type=str
    )
    parser.add_argument(
        '-o',
        '--output',
        action='store',
        dest='output_file',
        default='tmp_despiked_data.csv',
        help='path and filename of file to save results to',
        type=str
    )
    parser.add_argument(
        '-t',
        '--threshold',
        dest='threshold',
        default=3,
        help='z-score value threshold for flagging spikes (z-score > threshold)',
        type=float
    )
    parser.add_argument(
        '-w',
        '-window',
        dest='window',
        help='z-score sampling window length to estimate data median (seconds)',
        default=300.,
        type=float
    )
    args = parser.parse_args()
    Logger.info(f'loading data from: {args.input_file}')
    # Logger.info(f'writing data to: {args.output_file}')
    Logger.info(f'using a z-score threshold of {args.threshold}')
    Logger.info(f'using a sampling window of {args.window} sec to estimate the central value (median)')
    dt = pd.Timedelta(args.window, unit='seconds')
    # Load data
    df = pd.read_csv(args.input_file)
    # Populate Timestamp index from Epoch_UTC
    df.index = df.Epoch_UTC.apply(lambda x: pd.Timestamp(x*1e9))
    Logger.info('data loaded')
    df_out = pd.DataFrame()
    for _n, _c in enumerate(df.columns):
        if _c not in ['Epoch_UTC']:
            Logger.info(f'processing {_c}')
            s_ = df[_c].copy().rolling(dt).median()
            s_.index -= dt/2.
            s_Z = (df[_c] - s_)
            IND = np.abs(s_Z) <= args.threshold*s_Z.std()
            if _n == 0:
                df_out = df_out._append(df[_c][IND]).T
            else:
                df_out = pd.concat([df_out, df[_c][IND]], axis=1, ignore_index=False)
    # Write out updated Epoch_UTC timestamps from the Timestamp index
    df_out = df_out.assign(Epoch_UTC=[x.timestamp() for x in df_out.index])
    Logger.info(f'writing data to disk: {args.output_file}')
    df_out.to_csv(args.output_file, header=True, index=False)
    Logger.info('data written to disk - concluding main')


if __name__ == '__main__':
    ch = logging.StreamHandler()                                                            
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    Logger.addHandler(ch)
    Logger.setLevel(logging.INFO)
    main()