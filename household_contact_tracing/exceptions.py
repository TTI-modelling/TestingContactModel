# error handling module

from household_contact_tracing.branching_process_state import BranchingProcessState


class Error(Exception):
    """Base class for exceptions in this module"""
    pass


class ModelStateError(Error):
    """Exception raised when model is in an inappropriate state
    when a method or function is called.

    Args:
        state (BranchingProcessState): the state of the model when the error occurred.
        message (str): explanation of why the state was incorrect
    """

    def __init__(self, state: BranchingProcessState, message: str):
        self.state = state
        self.message = message
