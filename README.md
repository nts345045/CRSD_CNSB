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

1) Use `git` to clone this repository:  
```
git clone https://github.com/nts345045/CRSD_CNSB.git
cd CRSD_CNSB
```

2) Run everything (including setting up the conda environment and downloading the data repository):
```
bash run_everything.sh
```

Alternatively, `run_everything.sh` can be split out into the following steps
1) Create the necessary directory structures, install/activate the `conda` environment, and download data from MINDS@UW:
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
