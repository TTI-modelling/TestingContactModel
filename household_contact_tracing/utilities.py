from typing import Type, List
from copy import deepcopy
import itertools

from household_contact_tracing.branching_process_models import HouseholdLevelTracing


class ParameterError(Exception):
    """Raised if simulation parameters cannot be parsed."""


def run_parameterised_simulation(model_type: Type[HouseholdLevelTracing], num_steps: int,
                                 params: dict):
    """Assume sequence nesting is combinatorial at first."""
    processed_params = process_sequences(params)

    model_results = []
    for param_set in processed_params:
        model = model_type(param_set)
        model.run_simulation(num_steps, 1000)
        model_results.append(model)
    print(len(model_results))


def process_sequences(params: dict) -> List[dict]:
    if "sequences" not in params:
        processed_params = [params]
    else:
        if "nesting_type" in params["sequences"]:
            nesting_type = params["sequences"].pop("nesting_type")
            if nesting_type == "combination":
                processed_params = process_combinatorial_sequences(params)
            elif nesting_type == "linear":
                processed_params = process_linear_sequences(params)
            else:
                raise ParameterError(f"Unknown sequence nesting_type: '{nesting_type}'.\n"
                                     f"nesting_type must be 'linear' or 'combination'.")
        else:
            processed_params = process_linear_sequences(params)

    return processed_params


def process_linear_sequences(original_params: dict) -> List[dict]:
    """Take sequences of parameters and combine them linearly, returning the flattened
    parameter sets."""
    processed_params = []
    sequences = list(original_params["sequences"].values())
    if not validate_sequences_length(sequences):
        raise ParameterError(f"Invalid sequence parameters. To combine parameters linearly, all "
                             f"parameter sequences must be the same length.")

    for index in range(len(sequences[0])):
        param_set = deepcopy(original_params)
        del param_set["sequences"]
        for param in original_params["sequences"]:
            param_set[param] = original_params["sequences"][param][index]
        processed_params.append(param_set)
    return processed_params


def process_combinatorial_sequences(original_params: dict) -> List[dict]:
    """Take sequences of parameters and combine them """
    processed_params = []
    sequences = list(original_params["sequences"].values())
    parameter_names = list(original_params["sequences"])

    combinations = itertools.product(*sequences)

    for param_set in combinations:
        new_params = deepcopy(original_params)
        del new_params["sequences"]
        for name, value in zip(parameter_names, param_set):
            new_params[name] = value
        processed_params.append(new_params)

    return processed_params


def validate_sequences_length(sequences: List[list]):
    """Returns True if all lists in `sequences` are the same length. Else returns False."""
    for param in sequences:
        if len(param) != len(sequences[0]):
            return False
    return True
