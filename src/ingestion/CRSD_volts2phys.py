#!/usr/bin/python3
"""
:script: CRSD_volts2phys.py
:auth: Dougal D. Hansen, Nathan T. Stevens
:email: ddhansen3@wisc.edu, ntsteven@uw.edu
:license: CC-BY-4.0
:purpose:
    Convert raw output values in (milli)volts into physical units for
    sensors on the Cryogenic Ring Shear Device
:attribution:
    Original script written in MATLAB (RS_dataConversion_OSC_N.m) by D. Hansen in
    2022. Transcribed to Python and further annotated by N. Stevens in 2024.
    
varible explanations
--------------------
 - **raw_champer_postExp** (*millivolts*) -- raw data for the weights of the sample chamber, ice, and installed bed at the end of the experiment
 - **raw_postORING_torque** (*millivolts*) -- raw torque sensor measurement following drag test (i.e., chamber bolted to frame with no load applied)
 - **raw_OringTorque** (*millivolts*) -- raw data for O-ring drag, spinning the platen in an empty experimental chamber
 - **rawP1_zero** (*volts*) -- best estimate for a water pressure calibration for gauge 1
 - **rawP2_zero** (*volts*) -- best estimate for a water pressure calibration for gauge 2
 - **rawTorque_zero** (*millivolts*)
 - **rawTime** (*seconds*) -- time vector for experiment
 - **rawPw1** (*volts*) -- raw experimental values from pressure gauge 1
 - **rawPw2** (*volts*) -- raw experimental values from pressure gauge 2
 - **rawTorque** (*millivolts*) -- raw experiemntal values from the torque sensor
 - **rawRam** (*millivolts*) -- raw experimental values from the pressure transducer
"""
import os, logging, argparse
from scipy.io import loadmat
import numpy as np
import pandas as pd

# Create logger for module
Logger = logging.getLogger('CRSD_volts2phys.py')
ch = logging.StreamHandler()                                                            
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(fmt)
Logger.addHandler(ch)
Logger.setLevel(logging.INFO)

ROOT = os.path.join('..', '..', 'data')

def main(data_file):
    """main process
    Run processing converting volt-united measurements from CRSD transducers
    into kilopascal united measurements and timestamps from seconds relative
    to 1904-1-1 referenced times to epoch and UTC times (referenced to 1970-1-1)

    :param data_file: Raw CRSD *.mat data file path and name
    :type data_file: str
    :return: processed pressure data and timestamps
    :rtype: pandas.core.dataframe.DataFrame
    """    
    data_dict = loadmat(data_file)
    Logger.info('Running volts to physical units conversions for the UW-Madison CRSD Oscillatory Loading Experiment')
    Logger.info('data loaded')

    ## PHYSICAL CONSTANTS ##
    # Conversion coefficients
    psi2kpa = 6.895                     # [kPa/PSIG]
    # Timestamp conversion from 1904 reference to Epoch standard reference
    C_1904_to_epoch = (pd.Timestamp('1970-1-1') - pd.Timestamp('1904-1-1')).total_seconds()
    
    ## Sample Chamber Constants
    rout = 0.3                          # [m] Outer radius
    rin = 0.1                           # [m] Inner radius
    a_ring = np.pi*(rout**2. - rin**2.) # [m^2] area of the experimental chamber floor


    # Raw sample chamber load measurement pre experiment
    raw_chamber_preEXP = 0.109          # [mV] - single value recorded during experiment (lab notebook)

    ## EMPIRICAL CONSTANTS ##
    # Calibration coefficients
    C_torque = -14.34                   # Nm/mV
    C_torque_ram_correction = 0.243     # Nm/PSIG Peter Sobol's linear correction for applied ram pressure on torque sensor readings
    C_ram = 2000.                       # PSIG/mV
    C_ram_calib = 0.0231252             # PSIG -> kN calibration coefficient (From Peter Sobol)
    C_Pw = 50.                          # PSIG/V

    Logger.info('constants defined')

    ## LENEAR CORRECTION TERM CALCULATIONS ##
    # Zero Torque (with ice-filled sample chamber installed in frame)
    zero_torque = np.mean(data_dict['rawTorque_zero'])
    # Mean raw pressure transducer values
    zero_Pw1 = np.mean(data_dict['rawPw1_zero']) # [V]
    zero_Pw2 = np.mean(data_dict['rawPw2_zero']) # [V]
    # Raw sample chamber weight
    raw_chamber_postEXP = np.mean(data_dict['raw_chamber_postExp'])
    # Average of pre-/post-experiment mesurements
    chamber_weight = (raw_chamber_preEXP - raw_chamber_postEXP)/2. 
    # Raw O-ring drag
    Oring = np.mean(data_dict['rawOringTorque']) - np.mean(data_dict['raw_postORING_torque'])

    Logger.info('linear correction terms calculated')

    ## EFFECTIVE PRESSURE CONVERSION ##
    # Convert raw ram pressure to kPa
    Ram_PSIG = C_ram * data_dict['rawRam']
    # Convert ram pressure to axial load in kilonewtons
    Load_kN = (Ram_PSIG - C_ram*chamber_weight)*C_ram_calib
    # Convert axial load to applied normal load
    SigmaN_kPa = Load_kN/a_ring
    
    Logger.info('ram mV -> effective pressure kPa complete')

    ## WATER PRESSURE CONVERSION ##
    # Convert water pressure signals to kPa
    Pw1_kPa = C_Pw * psi2kpa * (data_dict['rawPw1']- zero_Pw1)
    Pw2_kPa = C_Pw * psi2kpa * (data_dict['rawPw2']- zero_Pw2)

    Logger.info('pressure transducer mV -> water pressure kPa converted')

    ## TORQUE CONVERSION ##
    # Convert torque signals into kPa 
    Torque_Nm = C_torque * (data_dict['rawTorque'] - Oring - zero_torque)
    # Include Peter Sobol's correction for sensor dependence on axial load
    Torque_Nm -= C_torque_ram_correction*Ram_PSIG
    # Convert torque into shear stress
    Tau_kPa = (3./(2.*np.pi) * Torque_Nm / (rout**3 - rin**3) / 1000.)

    Logger.info('torque mV -> shear stress kPa converted')

    ## Time Vector to DateTimes
    # Convert time vector to UTC
    # datetimes = [pd.Timestamp("1904-01-01") + pd.Timedelta(x, unit='sec') for x in data_dict['rawTime'].flatten()]
    epoch_times = data_dict['rawTime'] - C_1904_to_epoch
    Logger.info('timestamp seconds -> UTC date times converted')
    df_processed = pd.DataFrame({'SigmaN_kPa': SigmaN_kPa.flatten(),
                                'Tau_kPa': Tau_kPa.flatten(),
                                'Pw1_kPa': Pw1_kPa.flatten(),
                                'Pw2_kPa': Pw2_kPa.flatten()},
                                index = epoch_times.flatten())#},
                                # index = epoch_times.flatten().astype('datetime64[s]'))
    df_processed.index.name = 'Epoch_UTC'
    df_processed = df_processed.sort_index()
    Logger.info('dataframe composed, concluding main()')

    return df_processed


## RUN PROCESS WITH COMMAND LINE INTERFACE
if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        prog='CRSD_volts2phys.py',
        description='Converts raw CRSD data from the raw voltage measurements into calibrated physical units and UTC epoch times')
    # Input file
    parser.add_argument(
        '-i',
        '--input',
        action='store',
        dest='input_file',
        default='./RS_RAWdata_OSC_N.mat',
        type=str,
        help='path and file name for input file')
    # Output file
    parser.add_argument('-o',
        '--output',
        action='store',
        dest='output_file',
        default='./PhysicalUnitsData__UTC_Timing.csv', 
        type=str,
        help='path and file name for output file')
    # Parse Arguments
    arguments = parser.parse_args()
    # Reiterate inputs and outputs to log
    Logger.info(f'source: {arguments.input_file}')
    # save_file = os.path.join(ROOT, '1_preprocessed', 'Timeseries', 'PhysicalUnitsData__UTC_Timing.csv')
    Logger.info(f'output: {arguments.output_file}')
    Logger.info('Start of script')
    # RUN MAIN #
    df_proc = main(arguments.input_file)
    # SAVE TO DISK #
    Logger.info('Writing to disk')
    df_proc.to_csv(arguments.output_file, header=True, index=True)
    Logger.info('End of script')

