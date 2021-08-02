# TTI-modelling / TestingContactModel

[![Build Status](https://github.com/TTI-modelling/TestingContactModel/actions/workflows/python-package.yml/badge.svg)](https://github.com/TTI-modelling/TestingContactModel/actions/workflows/python-package.yml)


This repository contains tools to perform simulated infection and contact tracing using a branching process model

## Introduction

This readme provides information for users wishing to use the code to run simulations. Additional documentation for
users wishing to modify or extend the code is provided in `README_developers.md`.

The sections below are:
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Copyright and Licensing](#copyright--licensing)

<!-- toc -->

## Repository Structure

Example python scripts are stored within the `examples` directory, see [`Requirements`](#requirements) and 
[`Usage`](#usage) sections to get these working. 

The `household_contact_tracing` directory contains all of the python package modules. Contained within this, 
the `behaviours` directory stores the different behaviours classes, each containing the various strategies used to 
implement simulation processes such as `contact_rate_reduction` and `isolation`. Each of these 'behaviours' belongs
either to the higher level `infection` or the `intervention` processes.
To add new or extend functionalities, see the [`Contributing`](#contributing) section.

The `schemas` directory contains JSON schemas used to validate the JSON parameter files, used to initialise each 
simulation run.

The `views` directory contains classes representing the different outputs available, such as graph or textual views.
To improve or add further views, see the [`Contributing`](#contributing) section.

```
.
├── docs
├── examples
├── household_contact_tracing
│   ├── behaviours
│   │   ├── infection
│   │   ├── intervention
│   ├── schemas
│   ├── views
├── temp
├── test

```

## Requirements

All scripts in this repository are written in Python, tested on Unix and Windows systems. The minimum 
Python requirement is Python 3.6, though the code has been developed and tested with Python 3.8.

## Installation

To run simulations, the code should be installed as a Python package. We reccomend installing in a virtual environment
to avoid any potential conflicts with your base Python.

# Using Anaconda Python
To install the required packages using a Conda environment and pip, run the following from the root directory of the
repository: 
```
conda env create -f env_household_contact_tracing.yml
conda activate household-contact-tracing
```

# Using regular Python
To install the required packages using venv and pip,run the following from the root directory of the repository:
```
# Create the environment
python -m venv venv
# Activate the environment
venv/bin/activate.bat       # for Windows
source venv/bin/activate    # for Unix/OSX
# Install packages
pip install -U pip
pip install -r requirements.txt
```
Note that the command to activate the venv environment is different for Windows and for Unix/OSX

## Usage

Some example scripts and Interactive Jupyter notebooks are provided in the `examples` folder.

## Research

### Abstract
We explore strategies of contact tracing, isolation of infected individuals and quarantine of exposed individuals to control the SARS-Cov-2 epidemic using a household-individual branching process model. The explicit presence of households allows for modelling of household quarantine, and improved estimation of the effects of contact tracing. A contact tracing process designed to take advantage of the household structure is implemented, to understand whether such a strategy could control the epidemic. We evaluate the effects of different strategies of contact tracing, isolation and quarantine, such as two-step tracing, backwards tracing, smartphone tracing apps, and whether to test before the propagation contact tracing attempts. Uncertainty in SARS-Cov-2 transmission dynamics and contact tracing processes is modelled using prior distributions. The primary model outcome is the effect on the growth rate and doubling times of the epidemic, in combination with different levels of social distancing. Models of uptake and adherence to quarantine are applied, as well as contact recall, and how these affect the dynamics of contact tracing are considered. We find that a household contact tracing strategy allows for some relaxation of social distancing measures; however, it is unable to completely control the epidemic in the absence of other measures. Effectiveness of contact tracing and isolation is sensitive to delays, so strategies to improve speed relative to transmission could improve epidemic control, but non-uptake, imperfect recall and non-adherence to isolation can erode effectiveness. Improvements to the case identification rate could greatly benefit contact tracing interventions of SARS-Cov-2. Further, we find that once the epidemic has become established, the extinction times are on the scale of years when there is a small relaxation of the UK lockdown and contact tracing is employed.

### Authors
Martyn Fyles and Elizabeth Fearon

## Copyright & Licensing
This software has been developed by the Martyn Fyles and Elizabeth Fearon from the 
[The London School of Hygiene & Tropical Medicine](https://www.lshtm.ac.uk/) 
and Ann Gledson and Peter Crowther from the [Research IT](https://research-it.manchester.ac.uk/) 
group at the [University of Manchester](https://www.manchester.ac.uk/).

(c) 2020-2021 The London School of Hygiene & Tropical Medicine and the University of Manchester.
Licensed under the MIT license, see the LICENSE file for details.
