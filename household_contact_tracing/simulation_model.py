from __future__ import annotations
from abc import ABC, abstractmethod
import os
from typing import TYPE_CHECKING

from household_contact_tracing.views.simulation_view import SimulationView
from household_contact_tracing.simulation_states import SimulationState, ReadyState, RunningState

if TYPE_CHECKING:
    from household_contact_tracing.network import Network
    from household_contact_tracing.infection import Infection


class SimulationModel(ABC):
    
    __ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):

        # Set observer lists
        self._observers_param_change = []
        self._observers_graph_change = []
        self._observers_state_change = []
        self._observers_step_increment = []
        self._observers_simulation_stopped = []

        # State
        self._state = ReadyState(self)

    @property
    def state(self) -> SimulationState:
        return self._state

    @property
    def root_dir(self) -> str:
        return self.__ROOT_DIR

    @abstractmethod
    def run_simulation(self, max_time: int, infection_threshold: int) -> None:
        """ Run the simulation until it stops (e.g times out, or too many infectious nodes) """

    @property
    @abstractmethod
    def network(self) -> Network:
        """Return the network object that holds the Nodes."""

    @property
    @abstractmethod
    def infection(self) -> Infection:
        """Return the Infection object."""

    def simulation_stopped(self):
        """ The has stopped running """
        self.notify_observers_simulation_stopped()

    def graph_changed(self):
        """ The graph has changed """
        self.notify_observers_graph_change()

    def completed_step_increment(self):
        """ Completed incrementing simulation by one step """
        self.notify_observers_step_increment()

    # Register observers

    def register_observer_param_change(self, observer: object):
        """ Register as observer for parameter changes

        Arguments:
            observer -- the object to be added to the param change observers list
        """
        if observer not in self._observers_param_change:
            self._observers_param_change.append(observer)

    def register_observer_graph_change(self, observer: SimulationView):
        """ Register as observer for changes in model graph (nodes/households network)

        Arguments:
            observer -- the object to be added to the graph change observers list
        """
        if observer not in self._observers_graph_change:
            self._observers_graph_change.append(observer)

    def register_observer_state_change(self, observer: SimulationView):
        """ Register as observer for changes in model state (e.g. running, extinct, timed-out)
        Arguments:
            observer -- the object to be added to the state change observers list
        """
        if observer not in self._observers_state_change:
            self._observers_state_change.append(observer)

    def register_observer_simulation_stopped(self, observer: SimulationView):
        """ Register as observer for when simulation stops
        Arguments:
            observer -- the object to be added to the simulation stopped observers list
        """
        if observer not in self._observers_simulation_stopped:
            self._observers_simulation_stopped.append(observer)

    def register_observer_step_increment(self, observer: SimulationView):
        """ Register as observer for increment in simulation
        Arguments:
            observer -- the object to be added to the increment observers list
        """
        if observer not in self._observers_step_increment:
            self._observers_step_increment.append(observer)

    # Remove observers
    def remove_observer_param_change(self, observer: SimulationView):
        """ Remove as observer for parameter changes """
        try:
            self._observers_param_change.remove(observer)
        except ValueError:
            pass

    def remove_observer_graph_change(self, observer: SimulationView):
        """ Remove as observer for graph changes """
        try:
            self._observers_graph_change.remove(observer)
        except ValueError:
            pass

    def remove_observer_state_change(self, observer: SimulationView):
        """ Remove as observer for changes in model state (e.g. running, extinct, timed-out) """
        try:
            self._observers_state_change.remove(observer)
        except ValueError:
            pass

    def remove_observer_simulation_stopped(self, observer: SimulationView):
        """ Remove as observer for when simulation stops """
        try:
            self._observers_simulation_stopped.remove(observer)
        except ValueError:
            pass

    def remove_observer_step_increment(self, observer: SimulationView):
        """ Remove as observer for increment in simulation """
        try:
            self._observers_step_increment.remove(observer)
        except ValueError:
            pass

    # Notify Observers
    def notify_observers_graph_change(self, modifier=None):
        """ Notify observer about changes in graph (nodes/households network) """
        for observer in self._observers_graph_change:
            if observer != modifier:
                observer.graph_change(self)

    def notify_observers_state_change(self, modifier=None):
        """ Notify observer about changes in model state (e.g. running, extinct, timed-out)  """
        for observer in self._observers_state_change:
            if observer != modifier:
                observer.model_state_change(self)

    def notify_observers_simulation_stopped(self, modifier=None):
        """ Notify observer about when simulation has stopped """
        for observer in self._observers_simulation_stopped:
            if observer != modifier:
                observer.model_simulation_stopped(self)

    def notify_observers_step_increment(self, modifier=None):
        """ Notify observer about  increment in simulation """
        for observer in self._observers_step_increment:
            if observer != modifier:
                observer.model_step_increment(self)
