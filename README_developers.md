# TTI-modelling / TestingContactModel

[![Build Status](https://github.com/TTI-modelling/TestingContactModel/actions/workflows/python-package.yml/badge.svg)](https://github.com/TTI-modelling/TestingContactModel/actions/workflows/python-package.yml)

This repository contains tools to perform simulated infection and contact tracing using a branching process model

## Introduction 
This readme provides additional documentation for developers looking to extend or modify the code.

The sections below are:
- [Testing](#testing)
- [Logging](#logging)
- [Software Design](#software-design)  
- [Contributing](#contributing)

<!-- toc -->

## Testing
A set of python pytest test files can be found in the `test` directory, and can be run from the shell 
(once the necessary requirements are installed) with the command: `pytest`

The `test_integration.py` file contains a number of integration tests which run typical simulation scenarios.
The rest of the tests are unit tests though coverage is limited at the moment.

## Logging
Output is handled by the logging library *loguru*. Loguru handles both printing to stdout and more detailed logging
to a log file. Loguru messages have a severity level which determines where they are output to. Messages with a 
severity of *debug* and up are printed to the log file and messages with a severity of *info* and up are printed to 
stdout. The upshot of this is that if you want a message to be logged but not output to stdout use the *debug* level 
and if you want a message to print to stdout use the *info* level.

```
from loguru import logging

logging.info("I will be printed to both stdout and the log file.")
logging.debug("I will be printed only to the log file.")
```

The setup of the logger is done in the `household_contact_tracing/__init__.py` file. This should not need adjusting 
unless you want to alter the setup of the log file. 

## Software Design

A UML diagram is included in the `docs` folder, which gives a high level view of the classes and patterns used to 
build the simulation code.

### Extending functionality

Use the `current_UML.png` file (found in the `docs` folder) to help you visualise where any new amendments will
fit in.  The following is intended as a guide, and assumes basic knowledge of OO software design.

#### Updating Behaviours

1. How to design a new model, with combination of existing infection / intervention behaviours

    Create a new branching process model (can be within `branching_process_models.py` or in a separate module). It
    must inherit from `BranchingProcessModel` or one of its sub-classes (currently all contained within
    `branching_process_model`.) A good idea would be to find a suitable 'sibling' `BranchingProcessModel` class to use as
    a template. Copy the sibling class and make any amendments to which behaviours are used. 
    Behaviours are selected in the `_initialise_infection()` and `_initialise_intervention()` functions. See the 
    `household_contact_tracing/behaviours` folder to see the available behaviours.
   
2. How to write new behaviours (extending existing behaviour classes)

    See the modules in the `household_contact_tracing/behaviours` folder for current behaviours. Each of these
    modules has an abstract base (highest level parent class that must be sub-classed and can't be directly 
    instantiated.)  Below the abstract parent class is a set of sub-classes. Either inherit directly from
    the abstract class, or one of its sub-classes in that module.  It may help to select a 'sibling' class 
    suitable to use as a template, and make a copy of that and amend it as required.
   
    Select your new behaviour in the model, using the `_initialise_infection()` or `_initialise_intervention()` 
    function as appropriate.
   
3. How to write new behaviours (creating a new behaviour)

    If the new required behaviour does not need to use inheritance to apply to different scenarios/contexts,
    it can be written as a method and added to the `Infection` or `Intervention` class directly (these classes 
    can be found within the `household_contact_tracing` folder, in `infection.py` and `intervention.py`).

    See the modules in the `household_contact_tracing/behaviours` folder for current behaviours. 
    It may help to select a 'sibling' module (.py) file to copy and use as a template for your new behaviour
    module / set of classes.
   
    Once written, add usage of the new behaviour into either the `Infection` or `Intervention` classes as appropriate:
    Add the new behaviour as a parameter to the constructor, to be used as required by that class. 
    In the model class (e.g. BranchingProcessModel within `branching_process_models.py`), select the new behaviour 
    using the `_initialise_infection()` or `_initialise_intervention()` function as appropriate, passing it to the 
    constructor.
   
#### Views

1. Add a new view and register it with the model, and add to the controller.
    * Look at the views in the `household_contact_tracing/views` folder to see how these are created, copying the most 
      appropriate one to use as a template. You will see that all views inherit from SimulationView
      (`views/simulation_view.py`).  The views also contain a copy of the model, and use this to register themselves as 
      observers of the model, signing up to events of interest, so that they can output any information back to the user.
  
    * Add the new view to the `BranchingProcessController` class (or other class inherited from `SimulationController). 
      See others views added in that Controller class, to look at examples.  
   
### Parameters
Parameters are provided to the model when it is initialised. Parameters are provided in the form of a dictionary. 
Each parameter provided has an endpoint somewhere in the code, usually in of the behaviour classes. All parameters
have default values specified in the classes. Care should be taken to match the parameter names with the code 
endpoints.

Parameters are parsed against a schema which determines whether the parameter values are of the correct data type and 
any other specified validation checks. The parameter schemas can be found in the `schemas` folder. There is one 
schema for each model type. Where models inherit from other models, their schemas inherit the parameter schema from 
that model also.

When adding a new parameter to the model, it should be added to the relevant schema. At 
the moment, parameters not specified in the schema are allowed but in the future they will not be. This change is 
waiting on a stable implementation of a Python parser for JSON Schema draft 2020-12, which will allow use of the
`unevaluated_properties` keyword. At the moment, if a parameter is mistyped, it will be silently ignored, and the 
default value used instead.
  
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.