#!/bin/bash
# AUTH: Nathan T. Stevens
# EMAIL: ntsteven@uw.edu
# LICENSE: CC-BY-4.0
# PURPOSE:
# This script runs the following steps
# 0) setup the conda environment and directory structures for data processing
#    using scripts/step0_setup_processing_repo.sh
# 1) run all data processing from data repository files using
#    scripts/step1_run_processing.sh
# 2) renders all figures in the main text of Stevens and others (accepted) using
#    scripts/step2_generate_plots.sh
#
# NOTE: This script expects that you have already downloaded and un-tar'd the data
# repository per steps 1 and 2 in the README.md


bash scripts/step0_setup_processing_repo.sh
bash scripts/step1_run_processing.sh
bash scripts/step2_generate_plots.sh
