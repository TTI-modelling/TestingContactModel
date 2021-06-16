import pathlib

import jsonschema
import yaml
from loguru import logger


class Parameterised:
    """
        Base class containing logic for loading, checking and assignment of parameters

        Methods
        -------
            load_yaml(path: pathlib.Path) -> dict:
                Open the yaml file for checking parameters
            validate_parameters(params: dict, schema_path: str)
                Create the starting infectives
            .

    """

    @staticmethod
    def load_yaml(path: pathlib.Path) -> dict:
        with open(path) as input_file:
            data = yaml.load(input_file, yaml.SafeLoader)
        return data

    @classmethod
    def validate_parameters(cls, params: dict, schema_path: str):
        """Validate a dictionary of parameters against a schema."""
        logger.debug("Validating input parameters.")

        schema_path = pathlib.Path(schema_path)
        schema = cls.load_yaml(schema_path)

        schema_dir = schema_path.parent.absolute()

        resolver = jsonschema.RefResolver(schema_dir.as_uri(), None)
        jsonschema.validate(params, schema, resolver=resolver)
        logger.debug("Input parameters successfully validated.")

    def update_params(self, params: dict):
        """Update instance variables with anything in params."""
        if params:
            for param_name in self.__dict__:
                if param_name in params:
                    self.__dict__[param_name] = params[param_name]
