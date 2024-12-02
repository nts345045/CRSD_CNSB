"""
:script: processing/preprocess/despike_timeseries.py
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: This script removes large spikes from input
    time series data using a z-score metric
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
    df.index = pd.to_datetime(df.Epoch_UTC, unit='s')
    Logger.info('data loaded')
    # Get copy of dataframe, less the Epoch_UTC column
    df_tmp = df.copy()[df.columns.difference(['Epoch_UTC'])]
    # Apply rolling to all columns
    Logger.info('running rolling calculations')
    df_tmp_mea = df_tmp.copy().rolling(dt, center=True).median()
    df_tmp_std = df_tmp.copy().rolling(dt, center=True).std()
    Logger.info('Getting Z-scores')
    # Get deltas
    df_tmp_Z = df_tmp.copy() - df_tmp_mea
    # Get Z-scores
    df_tmp_Z /= df_tmp_std
    # Get unsigned Z-scores
    df_tmp_Z = df_tmp_Z.abs()
    # Filter out values greater than threshold
    df_tmp[df_tmp_Z > args.threshold] = np.nan
    Logger.info(f'Value counts filtered out for {len(df_tmp)} rows\n{(df_tmp_Z > args.threshold).sum()}')

    df_tmp = df_tmp.assign(Epoch_UTC=[x.timestamp() for x in df_tmp.index])
    Logger.info(f'writing data to disk: {args.output_file}')
    df_tmp.to_csv(args.output_file, header=True, index=False)
    Logger.info('data written to disk - concluding main')


if __name__ == '__main__':
    ch = logging.StreamHandler()                                                            
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    Logger.addHandler(ch)
    Logger.setLevel(logging.INFO)
    main()