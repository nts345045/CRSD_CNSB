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

DDIR='../data'
PDDIR='../processed_data'
RDIR='../results'
SRC='../src'

MODDIR="$PDDIR/steady_state"
EXP_TIMES='../param/UTC_experiment_times.csv'

CDIR='../processed_data/cavities'

FIGDIR='../results/figures'

if [ ! -d $DDIR ]; then
    echo "$DDIR does not exist - exiting"
    break
fi


echo "~~~~~~~~~~ GENERATING PUBLICATION FIGURES ~~~~~~~~~~"
# echo "Figure 1c - 3D render of experimental chamber"
# python $SRC/figures/JGLAC_Fig01a.py -o $FIGDIR -f 'png' -d 200

echo "Figure 2 - Steady State Parameter Space"
python $SRC/figures/JGLAC_Fig02.py -i $MODDIR -o $FIGDIR -f 'png' -d 200

# echo "Figure 3 - Effective Pressure Profile"
# python $SRC/figures/JGLAC_Fig03.py -i $STRESS_SMOOTH -o $FIGDIR -f 'png' -d 200

# echo "Figure 4 - Cavity geometry observation from T24"
# python $SRC/figures/JGLAC_Fig04.py -i $CDIR -o $FIGDIR -f 'png' -d 200

# echo "Figure 5 - Cavity geometry observation from T24"
# python $SRC/figures/JGLAC_Fig05.py -i $PDDIR -o $FIGDIR -f 'png' -d 200

# echo "Figure 6 - Cavity geometry observation from T06"
# python $SRC/figures/JGLAC_Fig06.py -i $PDDIR -o $FIGDIR -f 'png' -d 200

# echo "Figure 7 - Experiment T24 and T06 Crossplots"
# python $SRC/figures/JGLAC_Fig07.py -i $CDIR -o $FIGDIR -f 'png' -d 200

# echo "Figure 8 - Experiment T06 and T24 local stresses"
# python $SRC/figures/JGLAC_Fig08.py -i $CDIR -o $FIGDIR -f 'png' -d 200

# echo "Figure 9 - Comparison of T06 and T24 to common sliding rule parameters from Schoof (2005) parameterizations"
# python $SRC/figures/JGLAC_Fig09.py -i $PDDIR -o $FIGDIR -f 'png' -d 200