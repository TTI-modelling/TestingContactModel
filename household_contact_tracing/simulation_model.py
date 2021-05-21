from __future__ import annotations
from abc import ABC, abstractmethod
import os
from typing import TYPE_CHECKING

from household_contact_tracing.views.simulation_view import SimulationView
from household_contact_tracing.simulation_states import SimulationStateInterface, ReadyState, \
    RunningState, ExtinctState, TimedOutState, MaxNodesInfectiousState

if TYPE_CHECKING:
    from household_contact_tracing.network import Network

class SimulationModel(ABC):
    """
        Simulation Model
    """

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):

        # Set observer lists
        self._observers_param_change = []
        self._observers_graph_change = []
        self._observers_state_change = []
        self._observers_step_increment = []
        self._observers_simulation_stopped = []

        # States
        self._ready_state = ReadyState(self)
        self._running_state = RunningState(self)
        self._extinct_state = ExtinctState(self)
        self._timed_out_state = TimedOutState(self)
        self._max_nodes_infectious_state = MaxNodesInfectiousState(self)
        self._state = self._ready_state

    @property
    def state(self) -> SimulationStateInterface:
        return self._state

    @state.setter
    def state(self, state: SimulationStateInterface):
        self._state = state

    @property
    def ready_state(self) -> SimulationStateInterface:
        return self._ready_state

    @property
    def running_state(self) -> SimulationStateInterface:
        return self._running_state

    @property
    def extinct_state(self) -> SimulationStateInterface:
        return self._extinct_state

    @property
    def timed_out_state(self) -> SimulationStateInterface:
        return self._timed_out_state

    @property
    def max_nodes_infectious_state(self) -> SimulationStateInterface:
        return self._max_nodes_infectious_state

    @abstractmethod
    def run_simulation(self, max_time: int, infection_threshold: int) -> None:
        """ Run the simulation until it stops (e.g times out, or too many infectious nodes) """

    @property
    @abstractmethod
    def network(self) -> Network:
        """Return the network object that holds the Nodes."""

    def simulation_initialised(self):
        """ Initialise the simulation to starting values."""
        # NOTIFY OF STATE CHANGE
        self._state.initialise()

    def updated_parameters(self):
        """ Increment simulation by one step """
        self.notify_observer_param_change()

    def simulation_started(self):
        """ Start the simulation running."""
        # NOTIFY OF STATE CHANGE
        self._state.start_run()

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
