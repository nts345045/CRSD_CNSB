"""
:script: processing/preprocess/split_out_expriments.py
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: Apply one or more moving window average boxcar
    smoothing operators to 
"""

import argparse, logging, os
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
        '-f',
        '--timefile',
        dest='timefile',
        default=None,
        help='Target CSV file with 3 columns, "name", "starttime", and "endtime"'+\
             ' that define the UTC time bounds of oscillatory loading experiments\n'+\
             '"starttime" and "endtime" should be formatted such that pandas.read_csv '+\
             'can parse these fields as pandas.Timestamp objects',
        type=str
    )

    parser.add_argument(
        '-p',
        '-padding',
        dest='padding',
        default=21600,
        help='Seconds of extra data before and after specified experiment starttime and endtime values to include in output file(s)',
        type=float
    )

    parser.add_argument(
        '-o',
        '--output_path',
        action='store',
        dest='output_path',
        default='tmp_smoothed_data.csv',
        help='path for saving results. Filenames auto-assigned as '+\
             '<input_file_name>_<timefile_entry_name>.csv',
        type=str
    )

    parser.add_argument(
        '-t',
        '--data_type',
        action='store',
        dest='data_type',
        default='tmp',
        help='shorthand name of the data type being processed',
        type=str
    )

    #TODO: Migrate code form S3_smooth_despike_and_segment_data.py
    args = parser.parse_args()
    # 
    Logger.info(f'loading data from: {args.input_file}')    
    df_in = pd.read_csv(args.input_file)
    # Convert epoch into UTC Timestamp index
    df_in.index = df_in.Epoch_UTC.apply(lambda x: pd.Timestamp(x*1e9))

    Logger.info(f'loading timing information from: {args.timefile}')
    df_time = pd.read_csv(args.timefile, parse_dates=['starttime','endtime'], index_col=[0])

    dt_pad = pd.Timedelta(args.padding, unit='seconds')
    Logger.info(f'using a padding window of {args.padding/3600:.3f} hours ({args.padding:.1f} sec)')

    Logger.info(f'will write data with path and name prefix: {args.output_path}')
    


    # save_name = os.path.join(args.out_path,root_name,'_{seg_name}.csv')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        Logger.info(f'Creating directory: {args.output_path}')
    for name, row in df_time.iterrows():
        # Create subsetting index
        ind = (df_in.index >= row.starttime - dt_pad) & (df_in.index < row.endtime + dt_pad)
        # Subset data
        idf = df_in.copy()[ind]
        # Compose output file path and name
        out_fp = os.path.join(args.output_path,f'{name}-{args.data_type}.csv')
        Logger.info(f'writing output for {name}-{args.data_type} ({len(idf)} samples) to {out_fp}')
        # Write without index (get rid of temporary Timestamp index)
        idf.to_csv(out_fp, header=True, index=False)
    
    Logger.info('Concluding main() without errors')

if __name__ == '__main__':
    ch = logging.StreamHandler()                                                            
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    Logger.addHandler(ch)
    Logger.setLevel(logging.INFO)
    main()