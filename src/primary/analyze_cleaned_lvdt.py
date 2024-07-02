"""
:script: processing/primanry/analyze_cleaned_lvdt.py
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:license: CC-BY-4.0
:purpose: 
"""

import argparse, logging
import pandas as pd
import numpy as np

Logger = logging.getLogger('smooth_timeseries.py')

def main():
    parser = argparse.ArgumentParser(
        prog='smooth_timeseries.py',
        description='use a moving window z-score metric to remove anomalous spikes in data'
    )
    parser.add_argument(
        '-i',
        '--input',
        action='store',
        dest='input_file',
        default=None,
        help='path and filename of file to smooth',
        type=str
    )
    parser.add_argument(
        '-o',
        '--output',
        action='store',
        dest='output_file',
        default='tmp_smoothed_data.csv',
        help='path and filename of file to save results to',
        type=str
    )
    # TODO: Migrate core processes from processing/primary/S4_process_LVDT.py