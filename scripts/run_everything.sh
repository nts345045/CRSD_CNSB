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
EXP_TIMES='../param/UTC_experiment_times.csv'

# # Pressure Transducers Raw Data Ingestion
# STRESS_RAW_FILE="$RDDIR/RS_RAWdata_OSC_N.mat"
# STRESS_PUTC_FILE="$PPDIR/1_PhysicalUnitsData__UTC_Timing.csv"
# echo "~~~~~~~~~~ INGESTING TRANSDUCER TIMESERIES ~~~~~~~~~~"
python $SRC/ingestion/CRSD_volts2phys.py -i $STRESS_RAW_FILE -o $STRESS_PUTC_FILE

# # LVDT Raw Data Ingestion
LVDT_RAW_FILES="$RDDIR/LVDT/*.txt"
LVDT_UTC_FILE="$PPDIR/1_Stitched_LVDT_Data__UTC_Timing.csv"
echo "~~~~~~~~~~ INGESTING LVDT TIMESERIES ~~~~~~~~~~"
python $SRC/ingestion/merge_raw_LVDT.py -i $LVDT_RAW_FILES -o $LVDT_UTC_FILE -t 0.05 -r 1

# # Timeseries sampling homogenization
STRESS_RESAMPLED="$PPDIR/2_Resampled_Pressure_Data.csv"
LVDT_RESAMPLED="$PPDIR/2_Resampled_LVDT_Data.csv"
echo "~~~~~~~~~~ EVENLY SAMPLING TIMESERIES ~~~~~~~~~~"
python $SRC/preprocessing/evenly_sample_timeseries.py -i $STRESS_PUTC_FILE -o $STRESS_RESAMPLED -n 2
python $SRC/preprocessing/evenly_sample_timeseries.py -i $LVDT_UTC_FILE -o $LVDT_RESAMPLED -n 2

# Calculate effective pressure
python $SRC/preprocessing/calculate_effective_pressure.py -i $STRESS_RESAMPLED

# Timeseries despiking
STRESS_DESPIKED="$PPDIR/3_Despiked_Pressure_Data.csv"
LVDT_DESPIKED="$PPDIR/3_Despiked_LVDT_Data.csv"
echo "~~~~~~~~~~ DESPIKING TIMESERIES ~~~~~~~~~~"
python $SRC/preprocessing/despike_timeseries.py -i $STRESS_RESAMPLED -o $STRESS_DESPIKED
python $SRC/preprocessing/despike_timeseries.py -i $LVDT_RESAMPLED -o $LVDT_DESPIKED

# Timeseries smoothing
STRESS_SMOOTH="$PPDIR/4_Smoothed_Pressure_Data.csv"
LVDT_SMOOTH="$PPDIR/4_Smoothed_LVDT_Data.csv"
echo "~~~~~~~~~~ SMOOTHING TIMESERIES ~~~~~~~~~~"
python $SRC/preprocessing/smooth_timeseries.py -i $STRESS_DESPIKED -o $STRESS_SMOOTH -w 600 -s 75
python $SRC/preprocessing/smooth_timeseries.py -i $LVDT_DESPIKED -o $LVDT_SMOOTH -w 40 -s 5


# # Split Timeseries into Experiments
SPLIT_DATA_PATH="$PPDIR/5_split_data"
echo "~~~~~~~~~~ SPLITTING OUT TIMESERIES ~~~~~~~~~~"
python $SRC/preprocessing/split_out_experiments.py -i $STRESS_SMOOTH -o $SPLIT_DATA_PATH -t "Presssure" -f $EXP_TIMES
python $SRC/preprocessing/split_out_experiments.py -i $LVDT_SMOOTH -o $SPLIT_DATA_PATH -t "LVDT" -f $EXP_TIMES

# Cavity pick postprocessing
echo "~~~~~~~~~~ PROCESSING RAW CAVITY PICKS ~~~~~~~~~~"
CAVITY_RAW="$RDDIR/master_picks_T24_Exported_Points_v6.csv"
CAVITY_PP="$PPDIR/cavity_picks.csv"
python $SRC/preprocessing/process_cavity_pick_data.py -i $CAVITY_RAW -o $CAVITY_PP
# Process LVDT for empirical melt-rate calculations and getting extremum
 

# echo "Will run figure rendering scripts here"