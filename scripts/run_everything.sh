#!/bin/bash
#
# This script runs everything necessary to reproduce results presented
# in Stevens et al. (202X) from minimally processed data sources archived
# in their data repository.
#
# AUTHS: Nathan T. Stevens (ntsteven@uw.edu)
#        Dougal D. Hansen (ddhansen3@wisc.edu)
#        Peter E. Sobol (sobol@wisc.edu)
# 
# LICENSE: CC-BY 4.0
#

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
RDDIR='../data'
PPDIR='../processed_data'
SRC='../src'
# # Pressure Transducers Raw Data Ingestion
STRESS_RAW_FILE="$RDDIR/RS_RAWdata_OSC_N.mat"
STRESS_PUTC_FILE="$PPDIR/1_PhysicalUnitsData__UTC_Timing.csv"
python $SRC/ingestion/CRSD_volts2phys.py -i $STRESS_RAW_FILE -o $STRESS_PUTC_FILE

# # LVDT Raw Data Ingestion
LVDT_RAW_FILES="$RDDIR/LVDT/*.txt"
LVDT_UTC_FILE="$PPDIR/1_Stitched_LVDT_Data__UTC_Timing.csv"
python $SRC/ingestion/merge_raw_LVDT.py -i $LVDT_RAW_FILES -o $LVDT_UTC_FILE -t 0.05 -r 1

# # Timeseries sampling homogenization
STRESS_RESAMPLED="$PPDIR/2_Resampled_Pressure_Data.csv"
LVDT_RESAMPLED="$PPDIR/2_Resampled_LVDT_Data.csv"
python $SRC/preprocessing/evenly_sample_timeseries.py -i $STRESS_PUTC_FILE -o $STRESS_RESAMPLED
python $SRC/preprocessing/evenly_sample_timeseries.py -i $LVDT_UTC_FILE -o $LVDT_RESAMPLED

# # Timeseries despiking
STRESS_DESPIKED="$PPDIR/3_Despiked_Pressure_Data.csv"
LVDT_DESPIKED="$PPDIR/3_Despiked_LVDT_Data.csv"
python $SRC/preprocessing/despike_timeseries.py -i $STRESS_RESAMPLED -o $STRESS_DESPIKED
python $SRC/preprocessing/despike_timeseries.py -i $LVDT_RESAMPLED -o $LVDT_DESPIKED

# # Timeseries smoothing
STRESS_SMOOTH="$PPDIR/4_Smoothed_Pressure_Data.csv"
LVDT_SMOOTH="$PPDIR/4_Smoothed_LVDT_Data.csv"
python $SRC/preprocessing/smooth_timeseries.py -i $STRESS_DESPIKED -o $STRESS_SMOOTH -n 3 -w 10
python $SRC/preprocessing/smooth_timeseries.py -i $LVDT_DESPIKED -o $LVDT_SMOOTH -n 3 -w 10

# # Split Timeseries into Experiments

echo "Will run figure rendering scripts here"