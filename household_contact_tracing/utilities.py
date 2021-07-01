from typing import Type, List
from copy import deepcopy

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
    if "sequences" in params:
        processed_params = process_combinatorial_sequences(params)
    else:
        processed_params = [params]
    return processed_params


def process_combinatorial_sequences(original_params: dict) -> List[dict]:
    """Take sequences of parameters and combine them combinatorially returning the flattened
    parameter sets."""
    processed_params = []
    sequences = list(original_params["sequences"].values())
    if not validate_sequences(sequences):
        raise ParameterError(f"Invalid sequence parameters. All sequence parameters must be the "
                             f"same length.")

    for index in range(len(sequences[0])):
        param_set = deepcopy(original_params)
        del param_set["sequences"]
        for param in original_params["sequences"]:
            param_set[param] = original_params["sequences"][param][index]
        processed_params.append(param_set)
    return processed_params


def validate_sequences(sequences: List[list]):
    """Returns True if all lists in `sequences` are the same length. Else returns False."""
    for index, param in enumerate(sequences):
        if len(param) != len(sequences[0]):
            return False
    return True
