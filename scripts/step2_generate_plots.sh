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
SRC="$ROOT/src/figures"


echo "~~~~~~~~~~ GENERATING PUBLICATION FIGURES ~~~~~~~~~~"
# echo "Figure 1a - 3D render of sample chamber contents"
# python $SRC/JGLAC_Fig01a.py -o $FIGURE_DIR -f 'png' -d 200

# echo "Figure 2 - Steady State Parameter Space"
# python $SRC/JGLAC_Fig02.py -i $PD_MOD_DIR -o $FIGURE_DIR -f 'png' -d 200

# echo "Figure 3 - Effective Pressure Profile"
# python $SRC/JGLAC_Fig03.py -i $PD_TIME_DIR/4_Smoothed_Pressure_Data.csv -o $FIGURE_DIR -f 'png' -d 200

echo "Figure 4 - Cavity geometry observation from T24"
python $SRC/JGLAC_Fig04.py -e $PD_EXP_DIR -g $PD_GEOM_DIR -m $PD_MOD_DIR -o $FIGURE_DIR -f 'png' -d 200

# echo "Figure 5 - Cavity geometry observation from T24"
# python $SRC/JGLAC_Fig05.py -i $PDDIR -o $FIGURE_DIR -f 'png' -d 200

# echo "Figure 6 - Cavity geometry observation from T06"
# python $SRC/JGLAC_Fig06.py -i $PDDIR -o $FIGURE_DIR -f 'png' -d 200

# echo "Figure 7 - Experiment T24 and T06 Crossplots"
# python $SRC/JGLAC_Fig07.py -i $CDIR -o $FIGURE_DIR -f 'png' -d 200

# echo "Figure 8 - Experiment T06 and T24 local stresses"
# python $SRC/JGLAC_Fig08.py -i $CDIR -o $FIGURE_DIR -f 'png' -d 200

# echo "Figure 9 - Comparison of T06 and T24 to common sliding rule parameters from Schoof (2005) parameterizations"
# python $SRC/JGLAC_Fig09.py -i $PDDIR -o $FIGURE_DIR -f 'png' -d 200