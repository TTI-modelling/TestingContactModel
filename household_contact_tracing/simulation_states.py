from abc import ABC, abstractmethod


class SimulationState(ABC):
    """
        Simulation State Abstract class (State pattern)
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


class ReadyState(SimulationState):
    """
        Simulation is ready to start
    """
    name = 'ReadyState'
    allowed = ['RunningState']


class RunningState(SimulationState):
    """
        Simulation is running
    """
    name = 'RunningState'
    allowed = ['TimedOutState', 'ExtinctState', 'MaxNodesInfectiousState']


class ExtinctState(SimulationState):
    """
        Simulated outbreak has gone extinct
    """
    name = 'ExtinctState'
    allowed = ['RunningState']


class TimedOutState(SimulationState):
    """
        Simulated outbreak has timed out
    """
    name = 'TimedOutState'
    allowed = ['RunningState']


class MaxNodesInfectiousState(SimulationState):
    """
        Simulated outbreak has reached maximum number of infectious nodes
    """
    name = 'MaxNodesInfectiousState'
    allowed = ['RunningState']