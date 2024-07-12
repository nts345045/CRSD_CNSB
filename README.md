# CRSD_CNSB Code Respository  
This repository contains the source code used to produce results and figures presented in Stevens et al. (submitted) for the University of Wisconsin - Madison Cryogenic Ring Shear Device (CRSD) Cyclic N (effective pressure) Sinusoidal Bed (CNSB) experiment conducted in 2021.   

Manuscript in preparation for submission to the Journal of Glaciology


# License (CC-BY-4.0) 
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

2) Use the `setup_env.sh` script to generate a conda environment `crsd`:  
```
bash setup_env.sh
conda activate crsd
```

3) Get the input data for time-series processing into the repository root directory:  
 - Get data from temporary repository on [GoogleDrive](https://drive.google.com/file/d/1-QdYwzCuwoD8WaA8GHAhceLC2aZYDdHk/view?usp=share_link)  
 - un-tar the repository in the root directory of your cloned repository (`your/path/to/CRSD_CNSB`)
```
mv ~/Downloads/data.tar
tar -xf data.tar
rm data.tar
```

4) Run the `run_everything.sh` script from the `scripts` directory:  
```
cd scripts
bash run_everything.sh
```
This will take a few minutes to run and provides status updates to the command line.

NOTE: if your directory structure is different than described, you may need to change path settings in `run_everything.sh`
