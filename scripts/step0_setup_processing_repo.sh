#!/bin/bash
#
# This script sets up the directory structure, installs the conda environment, 
# and downloads the source data used in subsequent data processing (step1) and 
# plotting (step1) bash scripts. It should be run from the `scripts` directory.
#
# AUTHS: Nathan T. Stevens (ntsteven@uw.edu)
#        Dougal D. Hansen (ddhansen3@wisc.edu)
#        Peter E. Sobol (sobol@wisc.edu)
# 
# LICENSE: CC-BY-4.0
#

# Repository Root directory
ROOT='.'
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

ENVNAME='crsd'
ACTIVATE="$CONDA_PREFIX/bin/activate"


# Check if environment exists
if conda info --envs | grep -q "$ENVNAME"; then
    echo "Conda env '$ENVNAME' already exists"
# If it doesn't exist, create environment and install dependencies
else
    conda create --name $ENVNAME python=3.9 -y
    echo "~~~~~~~ Environment '$ENVNAME' created ~~~~~~~"
    source "$ACTIVATE" "$ENVNAME"
fi

if [[ "$CONDA_DEFAULT_ENV" == "$ENVNAME" ]]; then
    echo "Environment '$ENVNAME' already active"    
else
    source "$ACTIVATE" "$ENVNAME"
fi

if [[ "$CONDA_DEFAULT_ENV" == "$ENVNAME" ]]; then
    python -m pip install $ROOT
fi

# Set up .gitignored file structure
if [ ! -d $RAW_DIR ]; then
    echo "$RAW_DIR does not exist, creating directory"
    mkdir -p $RAW_DIR
fi

if [ ! -d $PD_DIR ]; then
    echo "$PD_DIR does not exist, creating directory"
    mkdir -p $PD_DIR
fi

if [ ! -d $PD_MOD_DIR ]; then
    echo "$PD_MOD_DIR does not exist, creating directory"
    mkdir -p $PD_MOD_DIR
fi

if [ ! -d $PD_TIME_DIR ]; then
    echo "$PD_TIME_DIR does not exist, creating directory"
    mkdir -p $PD_TIME_DIR
fi

if [ ! -d $PD_GEOM_DIR ]; then
    echo "$PD_GEOM_DIR does not exist, creating directory"
    mkdir -p $PD_GEOM_DIR
fi

if [ ! -d $PD_EXP_DIR ]; then
    echo "$PD_EXP_DIR does not exist, creating directory"
    mkdir -p $PD_EXP_DIR
fi

if [ ! -d $FIGURE_DIR ]; then
    echo "$FIGURE_DIR does not exist, creating directory"
    mkdir -p $FIGURE_DIR
fi

# Get data from MINDS@UW
curl -o $ROOT/data.zip "https://minds.wisconsin.edu/bitstream/handle/1793/89628/data.zip"
unzip $ROOT/data.zip
# mv data ..
# Cleanup
rm -r __MACOSX
rm data.zip