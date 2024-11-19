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

# Repository Root directory
ROOT='..'
# DATA DIRECTORY
RAW_DIR="$ROOT/data"
# PROCESSED DATA DIRECTORIES & SUBDIRECTORIES
PD_DIR="$ROOT/processed_data"
PD_MOD_DIR="$PD_DIR/model"
PD_TIME_DIR="$PD_DIR/timeseries"
PD_GEOM_DIR="$PD_DIR/geometry"
PD_EXP_DIR="$PD_DIR/experiments"
# RESULTS DIRECTORY
RESULTS_DIR="$ROOT/results"
FIGURE_DIR="$RESULTS_DIR/figures"
# SOURCE CODE DIRECTORY
SRC="$ROOT/src"

## Source Data Files
# Experiment Timeframes in UTC
EXP_TIMES="$RAW_DIR/UTC_experiment_times.csv"
LVDT_RAW="$RAW_DIR/LVDT/*.txt"
NT_RAW="$RAW_DIR/RS_RAWdata_OSC_N.mat"


# echo "!!!!!!!!!! STARTING TIME SERIES PROCESSING !!!!!!!!!!"
# # # Pressure Transducers Raw Data Ingestion
# STRESS_1_FILE="$PD_TIME_DIR/1_PhysicalUnitsData__UTC_Timing.csv"
# echo "~~~~~~~~~~ INGESTING TRANSDUCER TIMESERIES ~~~~~~~~~~"
# python $SRC/ingestion/CRSD_volts2phys.py -i $NT_RAW -o $STRESS_1_FILE

# # # LVDT Raw Data Ingestion
# LVDT_RAW_FILES="$RAW_DIR/LVDT/*.txt"
# LVDT_1_FILE="$PD_TIME_DIR/1_Stitched_LVDT_Data__UTC_Timing.csv"
# echo "~~~~~~~~~~ INGESTING LVDT TIMESERIES ~~~~~~~~~~"
# python $SRC/ingestion/merge_raw_LVDT.py -i $LVDT_RAW -o $LVDT_1_FILE -t 0.05 -r 1

# # # Timeseries sampling homogenization
# STRESS_2_FILE="$PD_TIME_DIR/2_Resampled_Pressure_Data.csv"
# LVDT_2_FILE="$PD_TIME_DIR/2_Resampled_LVDT_Data.csv"
# echo "~~~~~~~~~~ EVENLY SAMPLING TIMESERIES ~~~~~~~~~~"
# python $SRC/preprocessing/evenly_sample_timeseries.py -i $STRESS_1_FILE -o $STRESS_2_FILE -n 2
# python $SRC/preprocessing/evenly_sample_timeseries.py -i $LVDT_1_FILE -o $LVDT_2_FILE -n 2

# # Calculate effective pressure
# python $SRC/preprocessing/calculate_effective_pressure.py -i $STRESS_2_FILE

# # Timeseries despiking
# STRESS_3_FILE="$PD_TIME_DIR/3_Despiked_Pressure_Data.csv"
# LVDT_3_FILE="$PD_TIME_DIR/3_Despiked_LVDT_Data.csv"
# echo "~~~~~~~~~~ DESPIKING TIMESERIES ~~~~~~~~~~"
# python $SRC/preprocessing/despike_timeseries.py -i $STRESS_2_FILE -o $STRESS_3_FILE
# python $SRC/preprocessing/despike_timeseries.py -i $LVDT_2_FILE -o $LVDT_3_FILE

# # Timeseries smoothing
# STRESS_4_FILE="$PD_TIME_DIR/4_Smoothed_Pressure_Data.csv"
# LVDT_4_FILE="$PD_TIME_DIR/4_Smoothed_LVDT_Data.csv"
# echo "~~~~~~~~~~ SMOOTHING TIMESERIES ~~~~~~~~~~"
# python $SRC/preprocessing/smooth_timeseries.py -i $STRESS_3_FILE -o $STRESS_4_FILE -w 600 -s 75
# python $SRC/preprocessing/smooth_timeseries.py -i $LVDT_3_FILE -o $LVDT_4_FILE -w 40 -s 5

# # Split Timeseries into Experiments
# PADSEC=21600
# echo "~~~~~~~~~~ SPLITTING TIMESERIES INTO PERIODS ~~~~~~~~~~"
# python $SRC/preprocessing/split_out_experiments.py -i $STRESS_4_FILE -o $PD_EXP_DIR -t "Pressure" -f $EXP_TIMES -p $PADSEC
# python $SRC/preprocessing/split_out_experiments.py -i $LVDT_4_FILE -o $PD_EXP_DIR -t "LVDT" -f $EXP_TIMES -p $PADSEC

# echo "~~~~~~~~~~ PROCESSING LVDT DATA FOR MELT COEFFICIENTS ~~~~~~~~~~"
# python $SRC/primary/analyze_cleaned_lvdt.py -i $PD_EXP_DIR -o "$PD_EXP_DIR" -d $PADSEC

# echo "!!!!!!!!!! TIME SERIES PROCESSING COMPLETE !!!!!!!!!!"

# # Cavity pick postprocessing
# echo "~~~~~~~~~~ PROCESSING CAVITY GEOMETRIES ~~~~~~~~~~"
# python $SRC/preprocessing/process_cavity_picks.py -e $PD_EXP_DIR -r $RAW_DIR -o $PD_GEOM_DIR
# python $SRC/primary/analyze_cavity_geometry.py -e $PD_EXP_DIR -g $PD_GEOM_DIR -m $PD_MOD_DIR

# echo "!!!!!!!!!! CAVITY GEOMETRY PROCESSING COMPLETE !!!!!!!!!"

# Steady State Model Processing
# echo "~~~~~~~~~~ GENERATING PARAMETER SPACE FROM LLIBOUTRY/KAMB STEADY STATE THEORY ~~~~~~~~~~"
# python $SRC/primary/generate_parameter_space.py -o $PD_MOD_DIR -f "pandas" -p "UW" 

# Calculate shear stress correction for mounting bolt sockets' resisting stress
echo "~~~~~~~~~~ ESTIMATING SHEAR STRESS CORRECTIONS ~~~~~~~~~~"
python $SRC/primary/calculate_socket_resistance.py -e $PD_EXP_DIR -g $PD_GEOM_DIR

# echo "XXXXXXXXXXXX PROCESSING COMPLETE XXXXXXXXXXXX"