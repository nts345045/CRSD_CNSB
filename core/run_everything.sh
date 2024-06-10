#!/bin/bash
#
# This script runs everything necessary to reproduce results presented
# in Stevens et al. (202X) from minimally processed data sources archived
# in their data repository.
echo "Will create the environment"
echo "Will launch the environment"

echo "Will fetch data with curl here"

STRESS_RAW_FILE='../data/0_raw/Timeseries/RS_RAWdata_OSC_N.mat'
STRESS_PUTC_FILE='../data/1_preprocessed/Timeseries/PhysicalUnitsData__UTC_Timing.csv'
LVDT_RAW_FILES='../data/0_raw/Timeseries/LVDT/*.txt'
echo "Will run preprocessing scripts here"
python ../processing/preprocessing/CRSD_volts2phys.py -i $STRESS_RAW_FILE -o $STRESS_PUTC_FILE
echo "Will run processing scripts here"
echo "Will run figure rendering scripts here"