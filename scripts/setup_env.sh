#!/bin/bash
# This script sets up the conda environment
# for this repository and activates it
#
# AUTH: Nathan T. Stevens
# EMAIL: ntsteven@uw.edu
# LICENSE: CC-BY 4.0

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
    python -m pip install ..
fi
