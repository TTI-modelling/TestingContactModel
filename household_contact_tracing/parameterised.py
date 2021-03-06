import pathlib
import jsonschema
import yaml
from loguru import logger


class Parameterised:
    """
        Base class containing logic for loading, checking and assignment of parameters using
        JSON schema validation
        Inherit from this base class if parameter checking and loading is required.

        Methods
        -------
            load_yaml(path: pathlib.Path) -> dict
                Static method: Open the yaml file for checking parameters
            validate_parameters(cls, params: dict, schema_path: str)
                Class method: Create the starting infectives
            update_params(self, params: dict):
                Update instance variables with anything in params.
            get_param_value(self, param_name: str):
                Get the value of the parameter with param_name as key (if exists). If it doesn't exist, returns None

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

    def get_param_value(self, param_name: str):
        """ Get the value of the parameter with param_name as key (if exists)
            If it doesn't exist, returns None
        """
        if self.params and param_name in self.params:
            return self.params[param_name]
