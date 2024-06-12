#!/bin/bash
#
# This script runs everything necessary to reproduce results presented
# in Stevens et al. (202X) from minimally processed data sources archived
# in their data repository.
ENVNAME='crsd'
# Check if the environment is active
if [[ "$CONDA_DEFAULT_ENV" == "$ENVNAME" ]]; then
    echo "Environment '$ENVNAME' already active"
# If not active, but exists, activate
# elif conda info --envs | grep -q "$ENVNAME"
else
    echo "Activating '$ENVNAME'"
    conda activate "$ENVNAME"
fi
# Otherwise 
# else
#     echo "Environment '$ENVNAME' does not exist - exiting"
#     exit
# fi

PPDIR='../data/1_preprocessed'

# Pressure Transducers Initial Processing
STRESS_RAW_FILE='../data/0_raw/Timeseries/RS_RAWdata_OSC_N.mat'
STRESS_PUTC_FILE="'$PPDIR'/Timeseries/PhysicalUnitsData__UTC_Timing.csv"
#python ../processing/initial/CRSD_volts2phys.py -i $STRESS_RAW_FILE -o $STRESS_PUTC_FILE

# LVDT Initial Processing
LVDT_RAW_FILES='../data/0_raw/Timeseries/LVDT/*.txt'
LVDT_UTC_FILE="'$PPDIR'/Timeseries/Stitched_LVDT_Data__UTC_Timing.csv"
#python ../processing/initial/merge_raw_LVDT.py -i $LVDT_RAW_FILES -o $LVDT_UTC_FILE -t 0.05 -r 1

# Timeseries resampling
STRESS_RESAMPLED="'$PPDIR'/Timeseries/Resampled_Pressure_Data.csv"
LVDT_RESAMPLED="'$PPDIR'/Timeseries/Resampled_LVDT_Data.csv"
#python ../processing/preprocess/evenly_sample_timeseries.py -i $STRESS_PUTC_FILE -o $STRESS_RESAMPLED
#python ../processing/preprocess/evenly_sample_timeseries.py -i $LVDT_UTC_FILE -o $LVDT_RESAMPLED

# Timeseries despiking
STRESS_DESPIKED="'$PPDIR'/Timeseries/Despiked_Pressure_Data.csv"
LVDT_DESPIKED="'$PPDIR'/Timeseries/Despiked_LVDT_Data.csv"
python ../processing/preprocess/despike_timeseries.py -i $STRESS_RESAMPLED -o $STRESS_DESPIKED
python ../processing/preprocess/despike_timeseries.py -i $LVDT_RESAMPLED -o $LVDT_DESPIKED

# Timeseries smoothing
STRESS_SMOOTH="'$PPDIR'/Timeseries/Smoothed_Pressure_Data.csv"
LVDT_SMOOTH="'$PPDIR'/Timeseries/Smoothed_LVDT_Data.csv"

echo "Will run figure rendering scripts here"