"""
:module: JGLAC_Fig09.py
:version: 1 - Revision for JOG-2024-0083 (Journal of Glaciology)
:short ref: Stevens and others - Experimental constraints on transient glacier slip with ice-bed separation
:Figure #: 9 (initially submitted as Fig08)
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: Plot comparison of local stresses and area averaged stresses for experiments T06 and T24
"""
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    
    # Map Experimental Data Files
    T24_DAT = os.path.join(args.gpath,'EX_T24_cavity_metrics.csv')
    T06_DAT = os.path.join(args.gpath,'EX_T06_cavity_metrics.csv')

    # LOAD EXPERIMENTAL DATA #
    df_24 = pd.read_csv(T24_DAT)
    df_06 = pd.read_csv(T06_DAT)
    # RECONSTITUDE DATETIME INDEX
    df_24.index = pd.to_datetime(df_24.Epoch_UTC, unit='s')
    df_06.index = pd.to_datetime(df_06.Epoch_UTC, unit='s')
    # Define Reference time for T24 and T06
    t0_T24 = pd.Timestamp('2021-10-26T18:58')
    t0_T06 = pd.Timestamp('2021-11-1T16:09:15')


    # CALCULATE SIGMA_LOC
    o_loc_24 = df_24['N kPa'].values/df_24['S tot'].values
    o_loc_06 = df_06['N kPa'].values/df_06['S tot'].values


    fig = plt.figure(figsize=(7.5,7.5))
    GS = fig.add_gridspec(ncols=1,nrows=2,wspace=0)
    axs = [fig.add_subplot(GS[ii]) for ii in range(2)]

    # (a) \sigma_{loc} (t) & N(t) for T24
    axs[0].plot((df_24.index - t0_T24).total_seconds()/3600,\
                df_24['N kPa'].values*1e-3,'k-',label='$N$')

    axs[0].plot((df_24.index - t0_T24).total_seconds()/3600,\
                o_loc_24*1e-3,'-',color='blue',label='$\\sigma_{loc}$')
    axs[0].legend(loc='upper right')
    axs[0].set_ylabel('Pressure (MPa)')
    axs[0].set_xlabel('Elapsed Time During Exp. T24 [hr]')
    axs[0].set_xlim([-4, 121])
    axs[0].set_xticks(np.arange(0,132,12))
    ### (b) \sigma_{loc} (t) & N(t) for T06
    axs[1].plot((df_06.index - t0_T06).total_seconds()/3600,\
                df_06['N kPa'].values*1e-3,'k-',label='$N$')

    axs[1].plot((df_06.index - t0_T06).total_seconds()/3600,\
                o_loc_06*1e-3,'-',color='blue',label='$\\sigma_{loc}$')
    axs[1].legend(loc='center left')
    # axs[1].yaxis.set_label_position('right')
    # axs[1].yaxis.set_ticks_position('right')
    axs[1].set_ylabel('Pressure [MPa]')#,rotation=270,labelpad=15)
    axs[1].set_xlabel('Elapsed Time During Exp. T06 [hr]')
    axs[1].set_xlim([-2, 32])
    axs[1].set_xticks(np.arange(0,33,3))
    # for j_ in range(2):
    #     axs[j_].set_xticks(np.arange(0,6,1))
    #     axs[j_].set_xlim([-0.5,5.5])
    #     axs[j_].set_xlabel('Cycle [No.]')




    lblkw = {'fontsize':14,'fontweight':'extra bold','fontstyle':'italic',\
            'ha':'center','va':'center'}

    for i_,lbl_ in enumerate(['a','b']):
        xlims = axs[i_].get_xlim()
        ylims = axs[i_].get_ylim()


        axs[i_].text(xlims[0] + (xlims[1] - xlims[0])*0.025,\
                    ylims[0] + (ylims[1] - ylims[0])*0.95,\
                            lbl_,**lblkw)



    if not args.render_only:
        if args.dpi == 'figure':
            dpi = 'figure'
        else:
            try:
                dpi = int(args.dpi)

            except:
                dpi = 'figure'
        if dpi == 'figure':
            savename = os.path.join(args.output_path, f'JGLAC_Fig09_fdpi.{args.format}')
        else:
            savename = os.path.join(args.output_path, f'JGLAC_Fig09_{dpi}dpi.{args.format}')
        if not os.path.exists(os.path.split(savename)[0]):
            os.makedirs(os.path.split(savename)[0])
        plt.savefig(savename, dpi=dpi, format=args.format)

    if args.show:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='JGLAC_Fig09.py',
        description='Comparison of area-averaged effective pressure and localized contact stress for experiments T24 and T06'
    )
    parser.add_argument(
        '-g',
        '--geometry_path',
        dest='gpath',
        default=os.path.join('..','..','processed_data','geometry'),
        help='path to processed data files for cavity geometries',
        type=str
    )
    parser.add_argument(
        '-o',
        '--output_path',
        action='store',
        dest='output_path',
        default=os.path.join('..','..','results','figures'),
        help='path for where to save the rendered figure',
        type=str
    )
    parser.add_argument(
        '-f',
        '-format',
        action='store',
        dest='format',
        default='png',
        choices=['png','pdf','svg'],
        help='the figure output format (e.g., *.png, *.pdf, *.svg) callable by :meth:`~matplotlib.pyplot.savefig`. Defaults to "png"',
        type=str
    )
    parser.add_argument(
        '-d',
        '--dpi',
        action='store',
        dest='dpi',
        default='figure',
        help='set the `dpi` argument for :meth:`~matplotlib.pyplot.savefig. Defaults to "figure"'
    )
    parser.add_argument(
        '-s',
        '--show',
        action='store_true',
        dest='show',
        help='if included, render the figure on the desktop in addition to saving to disk'
    )
    parser.add_argument(
        '-r',
        '--render_only',
        dest='render_only',
        action='store_true',
        help='including this flag skips saving to disk'
    )
    args = parser.parse_args()
    main(args)