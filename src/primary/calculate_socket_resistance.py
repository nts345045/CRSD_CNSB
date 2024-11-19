"""
:module: src/primary/calculate_socket_resistance.py
:version: 1 - Revision on manuscript JOG-2024-0083 (Journal of Glaciology)
:short ref: Stevens and others - Experimental constraints on transient glacier slip with ice-bed separation
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: Calculate the shear stress correction factor to
    account for additional drag from mounting bolt sockets
    open to the ice-bed interface during Exp. T24 and Exp. T06
"""
import os, argparse
import pandas as pd
import numpy as np


def main(args):

    # Shared parameters
    NT_fstring = os.path.join(args.epath,'EX_{name}-Pressure.csv')
    LV_fstring = os.path.join(args.epath,'EX_{name}-LVDT-reduced.csv')
    CM_fstring = os.path.join(args.gpath,'EX_{name}_cavity_metrics.csv')
    CG_file = os.path.join(args.gpath,'Postprocessed_Cavity_Geometries.csv')

    df_CG = pd.read_csv(CG_file)
    df_CG.index = pd.to_datetime(df_CG.Epoch_UTC, unit='s')
    delta_tau_results_file = os.path.join(args.epath,'Delta_Tau_Estimates.csv')
    # Define experiment start times in UTC
    t0 = {'T24': pd.Timestamp('2021-10-26T18:58'),
          'T06': pd.Timestamp('2021-11-1T16:09:15')}
    # Define experiment durations
    dt = {'T24': pd.Timedelta(120.01, unit='hour'),
          'T06': pd.Timedelta(30.01, unit='hour')}
    
    with open(delta_tau_results_file, 'w') as file:
        file.write('Experiment,tau prime kPa,tau calc kPa,Delta tau kPa\n')
        for name in ['T24','T06']:
            df_NT = pd.read_csv(NT_fstring.format(name=name))
            df_NT.index = pd.to_datetime(df_NT.Epoch_UTC, unit='s')
            df_LV = pd.read_csv(LV_fstring.format(name=name))
            df_LV.index = pd.to_datetime(df_LV.Epoch_UTC, unit='s')
            df_CM = pd.read_csv(CM_fstring.format(name=name))
            df_CM.index = pd.to_datetime(df_CM.Epoch_UTC, unit='s')

            mean_tau_prime = df_CM[df_CM.index > t0[name] + dt[name]]['T kPa'].mean()
            mean_tau_calc = df_CM[df_CM.index > t0[name] + dt[name]]['hat T kPa'].mean()
            D_tau = mean_tau_prime - mean_tau_calc
            file.write(f'{name},{mean_tau_prime},{mean_tau_calc},{D_tau}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='calculate_socket_resistance.py',
        description='calculate the additional shear stress arising from open sockets at the ice-bed interface'
    )
    parser.add_argument(
        '-e',
        '--experiments_dir',
        action='store',
        dest='epath',
        default='../processed_data/experiments',
        type=str,
        help='path to the "experiments" processed data directory'
    )
    parser.add_argument(
        '-g',
        '--geometry_dir',
        action='store',
        dest='gpath',
        default='../processed_data/geometry',
        type=str,
        help='path to the "geometry" processed data directory'
    )
    args = parser.parse_args()
    main(args)