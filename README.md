# TTI-modelling / TestingContactModel

[![Build Status](https://github.com/TTI-modelling/TestingContactModel/actions/workflows/python-package.yml/badge.svg)](https://github.com/TTI-modelling/TestingContactModel/actions/workflows/python-package.yml)
<!--
[![Build Status](https://travis-ci.org/UoMResearchIT/UoM_AQ_Data_Tools.svg?branch=testing)](https://travis-ci.org/UoMResearchIT/UoM_AQ_Data_Tools) -->

This repository contains tools to perform simulated infection and contact tracing using a branching process model

The sections below are:
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Research](#research)
  - [Abstract](#abstract)
  - [Authors](#authors)
- [Testing](#testing) 
  - [Unit testing](#unit-testing)  
- [Contributing](#contributing)
- [Copyright and Licensing](#copyright--licensing)

<!-- toc -->

## Repository Structure

Example scripts are stored within the `examples` directory. The `household_contact_tracing` 
directory contains the python package modules. The `schemas` directory 
contains JSON schemas used to validate the JSON parameters, used to initialise each run.
[Todo - in progress.]

```
.
├── examples
├── household_contact_tracing
│   ├── behaviours
│   ├── schemas
│   ├── temp
│   ├── views
├── test

```


## Requirements

The processing scripts in this repository are written in python, tested on unix and [Todo] systems.

We recommend using conda to import the required python libraries. Using standalone pip is untested.

The processing scripts are written in python3. To install the packages needed for these 
(using conda and pip) use this script: `conda env create -f env_household_contact_tracing.yml`. 
To activate this environment use `conda activate household-contact-tracing`.

## Usage

The example scripts are in the examples directory and are written in jupyter notebook.


## Research

### Abstract
We explore strategies of contact tracing, isolation of infected individuals and quarantine of exposed individuals to control the SARS-Cov-2 epidemic using a household-individual branching process model. The explicit presence of households allows for modelling of household quarantine, and improved estimation of the effects of contact tracing. A contact tracing process designed to take advantage of the household structure is implemented, to understand whether such a strategy could control the epidemic. We evaluate the effects of different strategies of contact tracing, isolation and quarantine, such as two-step tracing, backwards tracing, smartphone tracing apps, and whether to test before the propagation contact tracing attempts. Uncertainty in SARS-Cov-2 transmission dynamics and contact tracing processes is modelled using prior distributions. The primary model outcome is the effect on the growth rate and doubling times of the epidemic, in combination with different levels of social distancing. Models of uptake and adherence to quarantine are applied, as well as contact recall, and how these affect the dynamics of contact tracing are considered. We find that a household contact tracing strategy allows for some relaxation of social distancing measures; however, it is unable to completely control the epidemic in the absence of other measures. Effectiveness of contact tracing and isolation is sensitive to delays, so strategies to improve speed relative to transmission could improve epidemic control, but non-uptake, imperfect recall and non-adherence to isolation can erode effectiveness. Improvements to the case identification rate could greatly benefit contact tracing interventions of SARS-Cov-2. Further, we find that once the epidemic has become established, the extinction times are on the scale of years when there is a small relaxation of the UK lockdown and contact tracing is employed.

### Authors
Martyn Fyles Elizabeth Fearon

## Testing
A set of python pytest test files can be found in the `test` directory, and can be run from the shell 
(once the necessary requirements are installed) with the command: `pytest`

## Contributing

### Extending functionality

* 
  
### Improvements to code
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## Copyright & Licensing
'''todo: this is from the previous/template'''

This software has been developed by the XXX (https://xxx/) and [Research IT](https://research-it.manchester.ac.uk/) 
group at the [University of Manchester](https://www.manchester.ac.uk/) for an 
[YYY](https://www.yyy/) project.

(c) 2020-2021 University of Manchester and XXX.
Licensed under the GPL-3.0 license, see the file LICENSE for details.
