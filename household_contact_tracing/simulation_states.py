from abc import ABC


class SimulationState(ABC):
    """
        Simulation State Abstract class (State pattern)
        For all simulations (not just e.g. Branching process models)

        Attributes
        ----------
        name (str): name of the state (replicates subclass name)
        allowed (list): list of state names that can be entered, following from this current state
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
        """ Switch to new state """
        if state.name in self.allowed:
            self.__class__ = state
            self.info = state_info
            self._simulation_model.notify_observers_state_change()
        else:
            raise ValueError('Current:', self, ' => switching to', state.name, 'not possible.')

    def __str__(self):
        return self.name

    def __repr__(self):
        return {'name': self.name,
                'info': self.info}


class BranchingProcessState(SimulationState):
    """
        Branching Process simulation states Abstract class (State pattern)
        Included to allow other BranchingProcess states sub-classes
    """
    pass


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