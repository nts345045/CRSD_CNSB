import argparse, os, logging
import pandas as pd
import numpy as np

Logger = logging.getLogger('calculate_effective_pressure')

def main(args):
    Logger.info(f'Loading: {args.input_file}')
    df = pd.read_csv(args.input_file)
    if not all(e in df.columns for e in ['SigmaN_kPa','Pw1_kPa','Pw2_kPa']):
        raise KeyError('Missing required columns in target file')
    else:
        pass
    Logger.info(f'Passed checks for calculating effective pressure')
    df = df.assign(Pe_kPa= df['SigmaN_kPa'] - np.nanmean(df[['Pw1_kPa','Pw2_kPa']],axis=1))
    Logger.info(f'saving to same file {args.input_file}')
    df.to_csv(args.input_file, header=True, index=False)

if __name__ == '__main__':
    ch = logging.StreamHandler()                                                            
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    Logger.addHandler(ch)
    Logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        prog='calculate_effective_pressure',
        description='conduct an in-place calculation of effecitve pressure on a CSV file'
    )

    parser.add_argument(
        '-i',
        '--input',
        dest='input_file',
        action='store',
        default=None,
        help='target *.csv file with columns SigmaN_kPa, Pw1_kPa, and Pw2_kPa with synchronous timestamps',
        type=str
    )

    args = parser.parse_args()


    main(args)