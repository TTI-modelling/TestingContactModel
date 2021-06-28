import json
from abc import ABC


class BranchingProcessState(ABC):
    """
        Branching Process State.  Abstract class (follows 'State' design pattern)
        States representing the possible states of Branching Process simulation models

        Attributes
        ----------
            name (str): name of the state (replicates subclass name)
            allowed (list): list of state names that can be switched to if changing directly from current state
            info (dict): extra information about the state

        Methods
        -------
            switch(self, state, **state_info)
                switch to a new state (state) and store useful info about the new state (state_info)

    """

    name = ''
    allowed = []
    info = {}

    def __init__(self, simulation_model):
        self._simulation_model = simulation_model

    def switch(self, state, **state_info):
        """
        Switch to a new state if new state is allowed (for current state)
            (If not, raise ValueError)

            Parameters:
                state (BranchingProcessState): The new state to be switched to

            Returns:
                None
        """
        if state.name in self.allowed:
            self.__class__ = state
            self.info = state_info
            self._simulation_model.notify_observers_state_change()
        else:
            raise ValueError('Current:', self, ' => switching to', state.name, 'not possible.')

    def __str__(self):
        return json.dumps(
            {'name': self.name,
             'info': self.info})


class ReadyState(BranchingProcessState):
    """
        Simulation is ready to start
    """
    name = 'ReadyState'
    allowed = ['RunningState']


class RunningState(BranchingProcessState):
    """
        Simulation is running
    """
    name = 'RunningState'
    allowed = ['TimedOutState', 'ExtinctState', 'MaxNodesInfectiousState']


class ExtinctState(BranchingProcessState):
    """
        Simulated outbreak has gone extinct
    """
    name = 'ExtinctState'
    allowed = ['RunningState']


class TimedOutState(BranchingProcessState):
    """
        Simulated outbreak has timed out
    """
    name = 'TimedOutState'
    allowed = ['RunningState']


class MaxNodesInfectiousState(BranchingProcessState):
    """
        Simulated outbreak has reached maximum number of infectious nodes
    """
    name = 'MaxNodesInfectiousState'
    allowed = ['RunningState']