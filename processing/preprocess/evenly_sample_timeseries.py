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
    Logger.info(f'writing data to: {args.output_file}')
    df = pd.read_csv(args.input_file, parse_dates=['Time_UTC'], index_col=[0])
    Logger.info('data loaded')
    df = df.sort_index()
    dt = np.round(np.median(df.epoch.values[1:] - df.epoch.values[:-1]), decimals=0)
    Logger.info(f'data to be resampled at {dt} sec intervals')
    df = df.resample(pd.Timedelta(dt, unit='sec')).median()
    Logger.info('data resampled')
    df.to_csv(args.output_file, header=True, index=True)
    Logger.info('data written to disk - concluing main')


if __name__ == '__main__':
    ch = logging.StreamHandler()                                                            
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    Logger.addHandler(ch)
    Logger.setLevel(logging.INFO)
    main()
