import pathlib

import jsonschema
import yaml
from loguru import logger


def load_yaml(path: pathlib.Path) -> dict:
    with open(path) as input_file:
        data = yaml.load(input_file, yaml.SafeLoader)
    return data


def parse_parameters(params: dict, schema: dict):
    """Parse a dictionary of parameters against a schema."""
    logger.debug("Validating input parameters.")
    jsonschema.validate(params, schema)
    logger.debug("Input parameters successfully validated.")
