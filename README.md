# TTI-modelling / TestingContactModel

[![Build Status](https://github.com/TTI-modelling/TestingContactModel/actions/workflows/python-package.yml/badge.svg)](https://github.com/TTI-modelling/TestingContactModel/actions/workflows/python-package.yml)

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
- [Logging](#logging)
- [Design](#design)  
- [Contributing](#contributing)
- [Copyright and Licensing](#copyright--licensing)

<!-- toc -->

## Repository Structure

Example scripts are stored within the `examples` directory, see `Requirements` and `Usage` sections to get these working. 
The `household_contact_tracing` directory contains the python package modules. 

Contained within this, 
the `behaviours` directory stores the different behaviours classes, each containing the various strategies used to 
implement simulation processes such as `contact_rate_reduction` and `isolation`. To add new or extend functionalities, 
see the `Contributing` section.

The `schemas` directory contains JSON schemas used to validate the JSON parameter files, used to initialise each 
simulation run.

The `views` directory contains classes representing the different outputs available, such as graph or textual views.
To improve or add further views, see the `Contributing` section.

```
.
├── examples
├── household_contact_tracing
│   ├── behaviours
│   ├── schemas
│   ├── views
├── temp
├── test

```


## Requirements

The processing scripts in this repository are written in python, tested on unix and [Todo] systems.

We recommend using conda to import the required python libraries. Using standalone pip is untested.

The processing scripts are written in python3. To install the packages needed for these 
(using conda and pip) use this script: `conda env create -f env_household_contact_tracing.yml`. 
To activate this environment use `conda activate household-contact-tracing`.

## Usage

The example jupyter notebook scripts are in the examples directory and are written in jupyter notebook.

To get a simple example running:

The required parts are highlighted in the following shortened excerpt:

```python
import household_contact_tracing.branching_process_models as bpm
from household_contact_tracing.simulation_controller import BranchingProcessController


params = {'outside_household_infectivity_scaling': 0.7,
            'contact_tracing_success_prob': 0.0, # doesn't matter, no tracing
            'overdispersion': 0.32,
            'asymptomatic_prob': 0.2,
            'asymptomatic_relative_infectivity': 0.35,
            'infection_reporting_prob': 0,
            'contact_trace': False,
            'test_delay': 2,
            'contact_trace_delay': 1,
            'incubation_period_delay': 5,
            'symptom_reporting_delay': 1,
            'household_pairwise_survival_prob': 0.2,
            'do_2_step': False,                      # doesn't matter, no tracing
            'reduce_contacts_by': 0.3,
            'prob_has_trace_app': 0,                 # doesn't matter, no tracing
            'hh_propensity_to_use_trace_app': 1,     # doesn't matter, no tracing
            'test_before_propagate_tracing': True,   # doesn't matter, no tracing
            'starting_infections': 1, 
            'node_will_uptake_isolation_prob': 1,    # doesn't matter, no tracing
            'self_isolation_duration': 0,            # doesn't matter, no tracing
            'quarantine_duration': 0,                # doesn't matter, no tracing
            'transmission_probability_multiplier': 1, # this isn't really useable (I would argue for removing it)
            'propensity_imperfect_quarantine': 0,    # doesn't matter no tracing
            'global_contact_reduction_imperfect_quarantine': 0, # doesn't matter, no tracing

         }

# Create controller and add model, then run
controller = BranchingProcessController(bpm.HouseholdLevelTracing(params))
controller.run_simulation(10)

# Update parameters
params['infection_reporting_prob'] = 0.5
params['self_isolation_duration'] = 10

# Re initialise with new parameters and Re-run
controller = BranchingProcessController(bpm.HouseholdLevelTracing(params))
controller.run_simulation(10)

# Add further parameters

params['number_of_days_to_trace_backwards'] = 2
params['number_of_days_to_trace_forwards'] = 5
params['recall_probability_fall_off'] = 1
params['probable_infections_need_test'] = True

# Create new model type 
controller = BranchingProcessController(bpm.IndividualLevelTracing(params))
# Switch off a view (e.g. the timeline graph views)
controller.timeline_view.set_display(False)
controller.run_simulation(10)


```


## Research

### Abstract
We explore strategies of contact tracing, isolation of infected individuals and quarantine of exposed individuals to control the SARS-Cov-2 epidemic using a household-individual branching process model. The explicit presence of households allows for modelling of household quarantine, and improved estimation of the effects of contact tracing. A contact tracing process designed to take advantage of the household structure is implemented, to understand whether such a strategy could control the epidemic. We evaluate the effects of different strategies of contact tracing, isolation and quarantine, such as two-step tracing, backwards tracing, smartphone tracing apps, and whether to test before the propagation contact tracing attempts. Uncertainty in SARS-Cov-2 transmission dynamics and contact tracing processes is modelled using prior distributions. The primary model outcome is the effect on the growth rate and doubling times of the epidemic, in combination with different levels of social distancing. Models of uptake and adherence to quarantine are applied, as well as contact recall, and how these affect the dynamics of contact tracing are considered. We find that a household contact tracing strategy allows for some relaxation of social distancing measures; however, it is unable to completely control the epidemic in the absence of other measures. Effectiveness of contact tracing and isolation is sensitive to delays, so strategies to improve speed relative to transmission could improve epidemic control, but non-uptake, imperfect recall and non-adherence to isolation can erode effectiveness. Improvements to the case identification rate could greatly benefit contact tracing interventions of SARS-Cov-2. Further, we find that once the epidemic has become established, the extinction times are on the scale of years when there is a small relaxation of the UK lockdown and contact tracing is employed.

### Authors
Martyn Fyles Elizabeth Fearon

## Testing
A set of python pytest test files can be found in the `test` directory, and can be run from the shell 
(once the necessary requirements are installed) with the command: `pytest`


## Logging
[Todo]

## UML Design

![Current UML design](./docs/current_uml.png "Current UML: Last updated 17/06/2021")

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
