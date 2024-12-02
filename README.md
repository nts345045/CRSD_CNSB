# CRSD_CNSB Code Respository  
This repository contains the source code used to produce results and figures presented in Stevens et al. (accepted) for the University of Wisconsin - Madison Cryogenic Ring Shear Device (CRSD) Cyclic N (effective pressure) Sinusoidal Bed (CNSB) experiment conducted in 2021.   

Manuscript accepted in the Journal of Glaciology (JOG-0083-2024-R2)


# License (CC-BY 4.0) 
Codes in this repository are provided as-is under the terms of the attached Creative Commons By Attribution license (CC-BY-4.0).

# Authors  
Nathan T. Stevens  
    - Pacific Northwest Seismic Network, University of Washington    
    - Department of Geoscience, University of Wisconsin-Madison  
Dougal D. Hansen  
    - Department of Geoscience, University of Wisconsin-Madison  
Lucas K. Zoet  
    - Department of Geoscience, University of Wisconsin-Madison  
Peter E. Sobol  
    - Department of Geoscience, University of Wisconsin-Madison  
Neal E. Lord  
    - Department of Geoscience, University of Wisconsin-Madison  

# Using this repository:  

To get all the data and run all the processing required to reproduce results presented in our manuscript, follow the steps below:  

0) Download a distribution of [conda](https://docs.anaconda.com/miniconda/miniconda-install/) for your operating system.  

1) Use `git` to clone this repository from GitHub. E.g.,  
```
git clone https://github.com/nts345045/CRSD_CNSB.git
cd CRSD_CNSB
```

2) Get the input data for time-series processing into the repository root directory:  
 - Get data from temporary repository on [GoogleDrive](https://drive.google.com/file/d/1-QdYwzCuwoD8WaA8GHAhceLC2aZYDdHk/view?usp=share_link)  
 - un-tar the repository in the root directory of your cloned repository (`your/path/to/CRSD_CNSB`)
```
mv ~/Downloads/data.tar
tar -xf data.tar
rm data.tar
```
 - you should have the following files / directories in a `data` subdirectory:
    - LVDT/
        - 211004_135802.txt
        - ...
        - 211117_171018.txt
    - master_picks_T24_Exported_Points.csv
    - RS_RAWdata_OSC_N.mat
    - UTC_experiment_times.csv

3) Run all processing (including setting up the conda environment)
```
bash run_everything.sh
```

Alternatively, `run_everything.sh` can be split out into the following steps
1) Create the necessary directory structures and install/activate the `conda` environment:
```
cd scripts
bash step0_setup_processing_repo.sh
```
2) Run data processing from the `scripts/` subdirectory
```
bash step1_run_processing.sh
```
3) Generate figures for the manuscript from the `scripts/` subdirectory
```
bash step2_generate_plots.sh
```
