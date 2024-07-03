import logging, os, argparse
import numpy as np
import pandas as pd
from src.model.lliboutry_kamb_model import calc_TS_single
from src.model.util import defined_experiment

Logger = logging.getLogger('generate_parameter_space.py')

def main(args):
    
    Logger.info(f'loading profile {args.profile}')
    profile = defined_experiment(args.profile)
    
    if args.save_format == 'numpy':
        save_file = f'{args.output_file}.npy'
    # elif args.format == 'pandas':
    #     save_file = f'{args.output_file}.csv'

    Nv = np.linspace(args.Nmin, args.Nmax, args.nN)
    Uv = np.linspace(args.Umin, args.Umax, args.nU)

    output_array = np.full(
        shape=(len(Nv), len(Uv), 6),
        fill_value=np.nan)
    
    Logger.info(f'starting grid sweep for space: {output_array.shape}')
    for ii, N_ in enumerate(Nv):
        if args.verbose:
            Logger.info(f'N iteration {ii+1}/{len(Nv)}')
        for jj, U_ in enumerate(Uv):
            output_array[ii,jj,:] = calc_TS_single(N_, U_, npts=args.ngx, **profile)

    path = os.path.split(args.output_file)[0]
    if not os.path.exists(path):
        Logger.warning(f'creating save path: {path}')
        os.makedirs(path)
    if args.save_format == 'numpy':
        Logger.info(f'saving to: {save_file}')
        np.save(save_file, output_array, allow_pickle=False)
    elif args.save_format == 'pandas':
        Logger.info('Formatting output for CSV by parameter type')
        rows = [f'{e:.1f} kPa' for e in Nv/1e3]
        cols = Uv
        for kk, name in enumerate(['SigmaN_kPa','Slip_mpy','S_total','Tau_kPa','Phi','R_mea']):
            if 'kPa' in name:
                data = output_array[:,:,kk]/1e3
            else:
                data = output_array[:,:,kk]
            df = pd.DataFrame(data=data,
                              index=rows,
                              columns=cols)
            df.columns.name='U_b m/yr'
            df.index.name='N kPa'
            Logger.info(f'Formatted dataframe for {name}')
            save_name = os.path.join(path, f'{name}_grid.csv')
            Logger.info(f'Saving as: {save_name}')
            df.to_csv(save_name, header=True, index=True, float_format='%.6f')



if __name__ == '__main__':
    ch = logging.StreamHandler()                                                            
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    Logger.addHandler(ch)
    Logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--output',
        action='store',
        dest='output_file',
        default='./parameter_space_values',
        help='file to save results to',
        type=str
    )

    parser.add_argument(
        '-f',
        '--format',
        action='store',
        dest='save_format',
        default='pandas',
        choices=['numpy','pandas'],
        help='choice of file save format. numpy -> *.npy bindary pandas -> *.csv',
        type=str
    )

    parser.add_argument(
        '-p',
        '--profile',
        action='store',
        dest='profile',
        default='UW',
        choices=['UW','ZI15'],
        help='indicate which bed/rheology profile to load',
        type=str
    )

    parser.add_argument(
        '-n',
        '--Nmin',
        action='store',
        dest='Nmin',
        default=1e5,
        help='minimum effective pressure for gridding. Defaults to 100000 Pa (100 kPa)',
        type=float
    )
    parser.add_argument(
        '-N',
        '--Nmax',
        action='store',
        dest='Nmax',
        default=9e5,
        help='minimum effective pressure for gridding. Defaults to 900000 Pa (900 kPa)',
        type=float
    )
    parser.add_argument(
        '-u',
        '--Umin',
        action='store',
        dest='Umin',
        default=0,
        help='minimum sliding velocity for gridding. Defaults to 0 m/year',
        type=float
    )
    parser.add_argument(
        '-U',
        '--Umax',
        action='store',
        dest='Umax',
        default=30,
        help='minimum effective pressure for gridding. Defaults to 100000 Pa (10 kPa)',
        type=float
    )
    parser.add_argument(
        '-x',
        '--N_points',
        action='store',
        dest='nN',
        default=1001,
        help='number of grid points for effective pressures. Defaults to 401',
        type=int
    )
    parser.add_argument(
        '-y',
        '--U_points',
        action='store',
        dest='nU',
        default=1001,
        help='number of grid points for sliding velocities. Defaults to 201',
        type=int
    )
    parser.add_argument(
        '-g',
        '--cavity_points',
        action='store',
        dest='ngx',
        default=5001,
        help='number of grid-points to use for modeling the '
    )

    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbose',
        action='store_true',
    )

    args = parser.parse_args()

    main(args)