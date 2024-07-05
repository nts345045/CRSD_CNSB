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
import matplotlib.pyplot as plt
from src.datetimeindex import *

Logger = logging.getLogger('analyze_cleaned_lvdt.py')


def main(args):
    
    f_L06 = glob.glob(os.path.join(args.input_path,'*T06-LVDT.csv'))
    f_L24 = glob.glob(os.path.join(args.input_path,'*T24-LVDT.csv'))
    if len(f_L06) == 1:
        f_L06 = f_L06[0]
    else:
        Logger.critical('multiple files found for Experiment T06 - quitting')
        sys.exit(1)
    if len(f_L24) == 1:
        f_L24 = f_L24[0]
    else:
        Logger.critical('multiple files found for Experiment T24 - quitting')
        sys.exit(1)
    
    df_L06 = pd.read_csv(f_L06)
    df_L06.index = pd.to_datetime(df_L06.Epoch_UTC, unit='s')
    df_L06 = df_L06.drop(['Epoch_UTC'], axis=1)
    Logger.info('Loaded experiment T06 data')
    df_L24 = pd.read_csv(f_L24)
    df_L24.index = pd.to_datetime(df_L24.Epoch_UTC, unit='s')
    df_L24 = df_L24.drop(['Epoch_UTC'], axis=1)
    Logger.info('Loaded experiment T24 data')

    melt_values = {}

    ### PICK  EXTREMUM ###
    # Create mask to remove edges included in earlier processing step
    DT_MASK = pd.Timedelta(args.padding,unit='second')
    INDSS = df_L24.index <= df_L24.index.min() + DT_MASK
    IND06 = (df_L06.index >= df_L06.index.min() + DT_MASK) & (df_L06.index <= df_L06.index.max() - DT_MASK)
    IND24 = (df_L24.index >= df_L24.index.min() + DT_MASK) & (df_L24.index <= df_L24.index.max() - DT_MASK)
    Logger.info('Picking LVDT extremum in experiments T24 and T06')
    T06_peaks = pick_extrema_indices(df_L06[IND06],T=pd.Timedelta(5,unit='hour'))
    T24_peaks = pick_extrema_indices(df_L24[IND24],T=pd.Timedelta(23,unit='hour'))
    ### CALCULATE STEADY STATE MELT RATE ###
    mSS = fit_dateline(df_L24[INDSS].index,df_L24[INDSS].values)
    Logger.info('The melt rate implied by steady-state is %.3e mm sec-1 (%.3e mm d-1)'%(mSS,mSS * 3600*24))
    melt_values.update({'steady_state': mSS[0]})

    ### CALCULATE MELT CORRECTIONS ###
    m06_M = fit_dateline(T06_peaks['I_max'][-3:],T06_peaks['V_max'][-3:])
    m06_m = fit_dateline(T06_peaks['I_min'][-3:],T06_peaks['V_min'][-3:])
    m06_u = np.mean([m06_m,m06_M])
    Logger.info('The melt rate being removed from T6 based on QSS is %.3e mm sec-1 (%.3e mm d-1)'%(m06_u,m06_u * 3600*24))
    melt_values.update({'T06': m06_u})
    S_L06r = reduce_series_by_slope(df_L06.copy(),m06_u,T06_peaks['I_min'][-3],T06_peaks['V_min'][-3])


    m24_M = fit_dateline(T24_peaks['I_max'][-2:],T24_peaks['V_max'][-2:])
    m24_m = fit_dateline(T24_peaks['I_min'][-2:],T24_peaks['V_min'][-2:])
    m24_u = np.mean([m24_m,m24_M])
    Logger.info('The melt rate being removed from T24 based on QSS is %.3e mm sec-1 (%.3e mm d-1)'%(m24_u,m24_u * 3600*24))
    melt_values.update({'T24': m24_u})
    S_L24r = reduce_series_by_slope(df_L24.copy(),m24_u,T24_peaks['I_min'][-2],T24_peaks['V_min'][-2])

    ### SAVE REDUCED TIMESERIES AND MELT VALUES ###
    S_melt = pd.Series(data=melt_values.values(), index=melt_values.keys(), name='melt factor mm/sec')
    df_L06r = pd.DataFrame(S_L06r)
    df_L06r = df_L06r.assign(Epoch_UTC=[x.timestamp() for x in S_L06r.index])
    df_L24r = pd.DataFrame(S_L24r)
    df_L24r = df_L24r.assign(Epoch_UTC=[x.timestamp() for x in S_L24r.index])

    if not os.path.exists(args.output_path):
        Logger.warning(f'creating output path: {args.output_path}')
        os.makedirs(args.output_path)

    for ipf, idf in [(f_L06, df_L06r), (f_L24, df_L24r)]:
        _, ifile = os.path.split(ipf)
        iname, _ = os.path.splitext(ifile)
        oname = f'{iname}-reduced.csv'
        Logger.info(f'Writing {oname} to disk in directory {args.output_path}')
        idf.to_csv(os.path.join(args.output_path,oname), header=True, index=False)
    
    Logger.info(f'Writing melt factor estimates to {args.output_path}')
    S_melt.to_csv(os.path.join(args.output_path,'melt_factor_estimates.csv'), header=True, index=True)
    if args.render_plots:
        ## Display results 
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=2,nrows=2)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[0,1])
        ax4 = fig.add_subplot(gs[1,1])

        ax1.plot(df_L24)
        ax1.plot(T24_peaks['I_min'],T24_peaks['V_min'],'bo')
        ax1.plot(T24_peaks['I_max'],T24_peaks['V_max'],'ro')
        ax2.plot(S_L24r)

        ax3.plot(df_L06)
        ax3.plot(T06_peaks['I_min'],T06_peaks['V_min'],'bo')
        ax3.plot(T06_peaks['I_max'],T06_peaks['V_max'],'ro')
        ax4.plot(S_L06r)

        plt.show()


if __name__ == '__main__':
    ch = logging.StreamHandler()                                                            
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    Logger.addHandler(ch)
    Logger.setLevel(logging.INFO)


    parser = argparse.ArgumentParser(
        prog='analyze_cleaned_lvdt.py',
        description='Conduct empirical melt factor analysis and corrections on LVDT data'
    )
    parser.add_argument(
        '-i',
        '--input_path',
        action='store',
        dest='input_path',
        default='.',
        help='path to smoothed, segmented LVDT data files',
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
        '-d',
        '--dt_padding',
        action='store',
        dest='padding',
        default=21600,
        help='amount of padding (in seconds) added to input LVDT data files',
        type=float
    )

    parser.add_argument(
        '-r',
        '--render_plots',
        action='store_true',
        dest='render_plots',
    )

    args = parser.parse_args()

    main(args)