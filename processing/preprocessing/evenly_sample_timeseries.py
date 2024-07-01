"""
:script: processing/preprocess/evenly_sample_timeseries.py
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: Synchronizes sampling rate of timeseries
    and interpolate into uniform temporal sampling
    with a sample spacing determined as the median
    inter-sample time for the timeseries data.
"""

import os, argparse, logging
import pandas as pd
import numpy as np

Logger = logging.getLogger('evenly_sample_timeseries.py')



def main():
    parser = argparse.ArgumentParser(
        prog='evenly_sample_timeseries.py',
        description='produce an evenly temporally sampled set of timeseries from experimetal data'
    )

    parser.add_argument(
        '-i',
        '--input',
        action='store',
        dest='input_file',
        default='',
        help='input CSV file path and name',
        type=str
    )
    parser.add_argument(
        "-o",
        '--output',
        action='store',
        dest='output_file',
        default='resampled_pressure_data.csv',
        help='output CSV file path and name',
        type=str
    )
    args = parser.parse_args()
    Logger.info(f'loading data from: {args.input_file}')
    # Logger.info(f'writing data to: {args.output_file}')
    # Read CSV
    df = pd.read_csv(args.input_file)
    # Update index as Timestamp objects
    df.index = df.Epoch_UTC.apply(lambda x: pd.Timestamp(x*1e9))
    Logger.info('data loaded')
    df = df.sort_index()
    # Get sampling period
    dt = np.round(np.median(df.Epoch_UTC.values[1:] - df.Epoch_UTC.values[:-1]), decimals=0)
    Logger.info(f'data to be resampled at {dt} sec intervals')
    # Resample
    df = df.resample(pd.Timedelta(dt, unit='second')).median()
    Logger.info('data resampled')
    # Update Epoch_UTC
    df.Epoch_UTC = [(ind - pd.Timestamp("1970-01-01T00:00:00")).total_seconds() for ind in df.index]
    Logger.info(f'writing to disk: {args.output_file}')
    df.to_csv(args.output_file, header=True, index=False)
    Logger.info('data written to disk - concluing main')


if __name__ == '__main__':
    ch = logging.StreamHandler()                                                            
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    Logger.addHandler(ch)
    Logger.setLevel(logging.INFO)
    main()
