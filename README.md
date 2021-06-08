# TTI-modelling / TestingContactModel

[![Build Status]()] # Todo complete this

This repository contains tools to perform simulated infection and contact tracing using a branching process model

The sections below are:
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Research](#research)
  - [Abstract](#abstract)
  - [Authors](#authors)
- [Testing](#testing) 
  - [Unit testing](#unit-testing)  
- [Developers](#developers)
- [Copyright and Licensing](#copyright--licensing)

<!-- toc -->

## Repository Structure

Operational scripts are stored within the `scripts` directory. Each of the subdirectories 
within this contains a README file guiding tool usage. The `environmental_data_modules` 
directory contains the python modules used by the tools. The `station_data` directory 
contains station specific metadata which has been collated for the datasets, these have 
been gathered from the MEDMI and AURN sources.

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

The example scripts are written in jupyter notebook and are in the examples directory
'''todo'''

## Research

### Abstract
We explore strategies of contact tracing, isolation of infected individuals and quarantine of exposed individuals to control the SARS-Cov-2 epidemic using a household-individual branching process model. The explicit presence of households allows for modelling of household quarantine, and improved estimation of the effects of contact tracing. A contact tracing process designed to take advantage of the household structure is implemented, to understand whether such a strategy could control the epidemic. We evaluate the effects of different strategies of contact tracing, isolation and quarantine, such as two-step tracing, backwards tracing, smartphone tracing apps, and whether to test before the propagation contact tracing attempts. Uncertainty in SARS-Cov-2 transmission dynamics and contact tracing processes is modelled using prior distributions. The primary model outcome is the effect on the growth rate and doubling times of the epidemic, in combination with different levels of social distancing. Models of uptake and adherence to quarantine are applied, as well as contact recall, and how these affect the dynamics of contact tracing are considered. We find that a household contact tracing strategy allows for some relaxation of social distancing measures; however, it is unable to completely control the epidemic in the absence of other measures. Effectiveness of contact tracing and isolation is sensitive to delays, so strategies to improve speed relative to transmission could improve epidemic control, but non-uptake, imperfect recall and non-adherence to isolation can erode effectiveness. Improvements to the case identification rate could greatly benefit contact tracing interventions of SARS-Cov-2. Further, we find that once the epidemic has become established, the extinction times are on the scale of years when there is a small relaxation of the UK lockdown and contact tracing is employed.

### Authors
Martyn Fyles Elizabeth Fearon

## Testing

## Developers


## Copyright & Licensing
'''todo: this is from the previous/template'''

This software has been developed by the XXX (https://xxx/) and [Research IT](https://research-it.manchester.ac.uk/) 
group at the [University of Manchester](https://www.manchester.ac.uk/) for an 
[YYY](https://www.yyy/) project.

(c) 2020-2021 University of Manchester and XXX.
Licensed under the GPL-3.0 license, see the file LICENSE for details.
