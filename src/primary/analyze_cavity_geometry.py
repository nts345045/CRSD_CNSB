import os, sys, argparse, logging
import pandas as pd
import numpy as np
from src.datetimeindex import pick_extrema_indices
import src.model.lliboutry_kamb_model as lkm

Logger = logging.getLogger('analyze_cavity_geometry.py')

def main(args):

    # Map Experimental Data Files from T24
    T24_NT = os.path.join(args.input_path,'5_split_data','EX_T24-Pressure.csv')
    T24_LV = os.path.join(args.input_path,'6_lvdt_melt_corrected','EX_T24-LVDT-reduced.csv')
    T24_SM = os.path.join(args.input_path,'cavity_picks.csv')

    # Map Experimental Data Files from T06
    T06_NT = os.path.join(args.input_path,'5_split_data','EX_T06-Pressure.csv')
    T06_LV = os.path.join(args.input_path,'6_lvdt_melt_corrected','EX_T06-LVDT-reduced.csv')


    # LOAD EXPERIMENTAL DATA #
    df_NT24 = pd.read_csv(T24_NT,parse_dates=True,index_col=[0])
    df_Z24 = pd.read_csv(T24_LV,parse_dates=True,index_col=[0])
    df_S24 = pd.read_csv(T24_SM,parse_dates=True,index_col=[0])
    df_NT06 = pd.read_csv(T06_NT,parse_dates=True,index_col=[0])
    df_Z06 = pd.read_csv(T06_LV,parse_dates=True,index_col=[0])

    df_NT24.index = pd.to_datetime(df_NT24.Epoch_UTC, unit='s')
    df_Z24.index = pd.to_datetime(df_Z24.Epoch_UTC, unit='s')
    df_S24.index = pd.to_datetime(df_S24.Epoch_UTC, unit='s')
    df_NT06.index = pd.to_datetime(df_NT06.Epoch_UTC, unit='s')
    df_Z06.index = pd.to_datetime(df_Z06.Epoch_UTC, unit='s')

    df_Z06 = df_Z06.resample(pd.Timedelta(3,unit='sec')).interpolate(method='from_derivatives')
    # SET REFERENCES/CONSTANTS
    hh_cl = 25.3*2  # [mm] Centerline bump height
    lbda_cl = 314.25
    hh_ow = 39.*2   # [mm] Outer experimental chamber bump height
    lbda_ow = 0.47123889803846897e3
    # Get offset models from observed reattachment and detachment point heights
    dRstoss = df_S24['Stoss dY POR mm'].values[-1]/hh_ow
    dRlee = df_S24['Lee dY PRO mm'].values[-1]/hh_ow
    dRmea = np.mean([dRstoss,dRlee])


    # Experiment Starting Times
    t0_T24 = pd.Timestamp('2021-10-26T18:58')
    t0_T06 = pd.Timestamp('2021-11-1T16:09:15')

    ### CALCULATE STEADY-STATE VALUES ASSUMING N(t) FORCING IS THE ONLY FREE VARIABLE
    # WITH EVERYTHING ELSE COMING FROM BED/CHAMBER GEOMETRY AND ZOET & IVERSON (2015)
    # EFFECTIVE VISCOSITY
    Logger.info('calculating steady-state model values')
    # Populate \tau(N) and S(N) vectors from steady-state model
    model_vectors = lkm.calc_parameter_space_from_NU(np.linspace(100e3,600e3,1001), 15)
    model_vectors = {'N_Pa': model_vectors[:,0], 'T_Pa': model_vectors[:,3]}
    # Populate estimates of cavity roof height
    SR_vectors = lkm.calc_geometry_space_from_NU(np.linspace(100e3,600e3,1001),15)
    SR_vectors = dict(zip(['Stot','Slee','Sstoss','Rmea','Rlea','Rstoss'],SR_vectors))



    dict_out = model_vectors.copy()
    dict_out.update(SR_vectors)
    dict_out = {k_:dict_out[k_] for k_ in set(list(dict_out.keys())) - set(['N_Pa'])}
    df_MOD = pd.DataFrame(dict_out,index=pd.Index(model_vectors['N_Pa'],name='N_Pa'))

    ### CONDUCT PROCESSING FOR T24 GEOMETRIES

    # Downsample N(t) and \tau(t) data to match sampling from LVDT data
    NPa_ds_T24 = np.interp((df_Z24.index - t0_T24).total_seconds(),\
                        (df_NT24.index - t0_T24).total_seconds(),\
                        df_NT24['Pe_kPa'].values)*1e3
    TPa_ds_T24 = np.interp((df_Z24.index - t0_T24).total_seconds(),\
                        (df_NT24.index - t0_T24).total_seconds(),\
                        df_NT24['T_kPa'].values)*1e3

    ## Get reference \\Delta y(t)
    IND = np.argmax(np.abs((df_Z24.index - t0_T24).total_seconds()))
    ## Calculate Reattachment Point Height Steady-State Reference Value
    DR0 = df_Z24['LVDT_mm red'].values[IND]/hh_cl



    # INTERPOLATE STEADY-STATE VALUES TO N(t) INPUTS
    T_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],model_vectors['T_Pa'])
    S_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],model_vectors['S'])

    # Calculate non-dimensionalized cavity height & contact length metrics
    # Assume mean of cavity height is representative for ice-bed separation
    R_mea_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Rmea'])
    S_tot_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Stot'])
    # Assume minimum cavity height (reattachment point) is representative for ice-bed separation
    R_stoss_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Rstoss'])
    S_stoss_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Sstoss'])
    # Assume maximum cavity height (detachment point) is representative for ice-bed separation
    R_lee_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Rlee'])
    S_lee_hat24 = np.interp(NPa_ds_T24,model_vectors['N_Pa'],SR_vectors['Slee'])



    ### Calculate observed cavity height metrics from LVDT + observed SS 
    # Calculate R(t) from stoss (reattachment) point
    Rstoss = (df_Z24['LVDT_mm red'].values/hh_cl) - DR0 + (1 + dRstoss)
    Sstoss = np.interp(Rstoss,SR_vectors['Rstoss'],SR_vectors['Sstoss'],period=24)
    # Calculate R(t) from lee (detachment) point
    Rlee = (df_Z24['LVDT_mm red'].values/hh_cl) - DR0 + (1 + dRlee)
    Slee = np.interp(Rlee,SR_vectors['Rlee'],SR_vectors['Slee'],period=24)
    # Calculate R(t) from average of lee and stoss points
    Rbar = (df_Z24['LVDT_mm red'].values/hh_cl) - DR0 +(1 + dRmea)
    Stot = np.interp(Rbar,SR_vectors['Rmea'],SR_vectors['Stot'],period=24)

    # CALCULATE DRAG FROM OBSERVED N(t) AND R(t) ASSUMING THE R2S MAPPING FROM STEADY STATE THEORY IS CORRECT
    # R2S_hat = np.interp(Rlee,df_MOD['Rmea'].values,df_MOD['S'].values)
    # R2S2tau_lee_hat = ssm.calc_from_SN(NPa_ds_T24,R2S_hat)
    # R2Sstoss_hat = np.interp(Rstoss,df_MOD['Rstoss'].values,df_MOD['Sstoss'].values)
    # R2Stot_hat = np.interp(Rbar,df_MOD['Rmea'].values,df_MOD['S'].values)


    df_summary_T24 = pd.DataFrame({'N kPa':NPa_ds_T24*1e-3,\
                                'T kPa':TPa_ds_T24*1e-3,'hat T kPa':T_hat24*1e-3,\
                                'R stoss':Rstoss,'hat R stoss':R_stoss_hat24,\
                                'R mea':Rbar,'hat R mea':R_mea_hat24,\
                                'R lee':Rlee,'hat R lee':R_lee_hat24,\
                                'S tot':Stot,'hat S tot':S_tot_hat24,\
                                'S lee':Slee,'hat S lee':S_lee_hat24,\
                                'S stoss':Sstoss,'hat S stoss':S_stoss_hat24},index=df_Z24.index)



    ## CONDUCT PROCESSING FOR T06 GEOMETRIES

    NPa_ds_T06 = np.interp((df_Z06.index - t0_T06).total_seconds(),\
                        (df_NT06.index - t0_T06).total_seconds(),\
                        df_NT06['Pe_kPa'].values)*1e3
    TPa_ds_T06 = np.interp((df_Z06.index - t0_T06).total_seconds(),\
                        (df_NT06.index - t0_T06).total_seconds(),\
                        df_NT06['T_kPa'].values)*1e3

    ## Get reference \\Delta y(t)
    IND = np.argmin(np.abs((df_Z06.index - t0_T06).total_seconds()))
    ## Calculate Reattachment Point Height Steady-State Reference Value
    DR0 = df_Z06['LVDT_mm red'].values[IND]/hh_cl

    # INTERPOLATE STEADY-STATE VALUES TO N(t) INPUTS
    T_hat6 = np.interp(NPa_ds_T06,model_vectors['N_Pa'],model_vectors['T_Pa'])
    S_hat6 = np.interp(NPa_ds_T06,model_vectors['N_Pa'],model_vectors['S'])

    # Calculate non-dimensionalized cavity height for a range of models
    # Assume mean of cavity height is representative for ice-bed separation
    R_mea_hat6 = np.interp(NPa_ds_T06,model_vectors['N_Pa'],SR_vectors['Rmea'])
    S_tot_hat6 = np.interp(NPa_ds_T06,model_vectors['N_Pa'],SR_vectors['Stot'])
    # Assume minimum cavity height (reattachment point) is representative for ice-bed separation
    R_stoss_hat6 = np.interp(NPa_ds_T06,model_vectors['N_Pa'],SR_vectors['Rstoss'])
    S_stoss_hat6 = np.interp(NPa_ds_T06,model_vectors['N_Pa'],SR_vectors['Sstoss'])
    # Assume maximum cavity height (detachment point) is representative for ice-bed separation
    R_lee_hat6 = np.interp(NPa_ds_T06,model_vectors['N_Pa'],SR_vectors['Rlee'])
    S_lee_hat6 = np.interp(NPa_ds_T06,model_vectors['N_Pa'],SR_vectors['Slee'])


    # Calculate Observed R(t) for a range of statistical representations
    # Calculate R(t) from stoss (reattachment) point
    Rstoss_T06 = (df_Z06['LVDT_mm red'].values/hh_cl) - DR0 + (1 + dRstoss)
    Sstoss_T06 = np.interp(Rstoss_T06,SR_vectors['Rstoss'],SR_vectors['Sstoss'],period=6)
    # Calculate R(t) from lee (detachment) point
    Rlee_T06 = (df_Z06['LVDT_mm red'].values/hh_cl) - DR0 + (1 + dRlee)
    Slee_T06 = np.interp(Rlee_T06,SR_vectors['Rlee'],SR_vectors['Slee'],period=6)
    # Calculate R(t) from average of lee and stoss points
    Rbar_T06 = (df_Z06['LVDT_mm red'].values/hh_cl) - DR0 + (1 + dRmea)
    Stot_T06 = np.interp(Rbar_T06,SR_vectors['Rmea'],SR_vectors['Stot'],period=6)

    df_summary_T06 = pd.DataFrame({'N kPa':NPa_ds_T06*1e-3,'T kPa':TPa_ds_T06*1e-3,'hat T kPa':T_hat6*1e-3,\
                                'R stoss':Rstoss_T06,'hat R stoss':R_stoss_hat6,\
                                'R mea':Rbar_T06,'hat R mea':R_mea_hat6,\
                                'R lee':Rlee_T06,'hat R lee':R_lee_hat6,\
                                'S tot':Stot_T06,'hat S tot':S_tot_hat6,\
                                'S lee':Slee_T06,'hat S lee':S_lee_hat6,\
                                'S stoss':Sstoss_T06,'hat S stoss':S_stoss_hat6},index=df_Z06.index)

    if not os.path.exists(args.output_dir):
        os.makediers(args.output_dir)

    ## Write Corrected & Modeled Heights to File
    R6_out = os.path.join(args.output_dir,'EX_T06_cavity_metrics.csv')
    R24_out = os.path.join(args.output_dir,'EX_T24_cavity_metrics.csv')
    Rmod_out = os.path.join(args.output_dir,'modeled_cavity_values.csv')
    df_summary_T06.to_csv(R6_out,header=True,index=True)
    df_summary_T24.to_csv(R24_out,header=True,index=True)
    df_MOD.to_csv(Rmod_out,header=True,index=True)


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
        default='processed_data',
        help='path to smoothed, segmented LVDT data files',
        type=str
    )
    parser.add_argument(
        '-o',
        '--output_path',
        action='store',
        dest='output_path',
        default='processed_data/cavity_geometry_parameters',
        help='path and filename of file to save results to',
        type=str
    )

    args = parser.parse_args()

    main(args)