# UoM AQ Data Tools

[![Build Status]()] # Todo complete this

This repository contains tools for obtaining and processing UK air quality data.

The sections below are:
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Research](#research)
- [Authors](#authors)

- [Testing](#testing) 
  - [Unit testing](#unit-testing)

<!-- toc -->

## Repository Structure

Operational scripts are stored within the `scripts` directory. Each of the subdirectories 
within this contains a README file guiding tool usage. The `environmental_data_modules` 
directory contains the python modules used by the tools. The `station_data` directory 
contains station specific metadata which has been collated for the datasets, these have 
been gathered from the MEDMI and AURN sources.

```
.
├── environmental_data_modules
├── scripts
│   ├── AURN_Data_Download
│   ├── Combine_Data
│   ├── Data_Imputation_Testing
│   ├── Data_Processing
│   ├── EMEP_Data_Extraction
│   └── MEDMI_Data_Download
└── station_data
```


## Requirements

The processing scripts in this repository are written in python, tested on unix and OSX
systems.

The EMEP (and WRF) models are written in fortran - follow the references below for compiling 
these, preparing model inputs, and performing the simulations.

The MEDMI dataset are accessed using the python2 installation on their system, no more
packages require installing to run the scripts for this.

We recommend using conda to import the required python libraries. Using standalone pip is
untested. 

The processing scripts for extracting the EMEP data are written in python3. To
install the packages needed for these (using conda and pip) use this script:
`conda env create -f env_emep.yml`.
To activate this environment use `conda activate emep`.

The processing scripts for obtaining the AURN dataset, and processing all datasets, are
written in python3. To install the packages needed for these, use this script: 
`conda env create -f env_aurn_medmi.yml`.
To activate this environment this `conda activate aurn_medmi`.




## Copyright & Licensing

# Todo: this is from the previous/template

This software has been developed by the [Research IT](https://research-it.manchester.ac.uk/) 
group at the [University of Manchester](https://www.manchester.ac.uk/) for an 
[Alan Turing Institute](https://www.turing.ac.uk/) project.

(c) 2019-2021 University of Manchester.
Licensed under the GPL-3.0 license, see the file LICENSE for details.
