from __future__ import annotations
from abc import ABC, abstractmethod
import os

from household_contact_tracing.simulation_states import SimulationState, BranchingProcessState, \
    ReadyState

from household_contact_tracing.network import Network


class SimulationModel(ABC):
    """
        An abstract base class used to represent simulation models at the highest level.

        Attributes
        ----------
        root_dir (str):
            root directory of containing package

        Methods
        -------

    """

    __ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):

        # Set observer lists
        self._observers_step_increment = []
        self._observers_simulation_stopped = []
        self._observers_state_change = []

    @property
    @abstractmethod
    def state(self) -> SimulationState:
        """ Get the current state of the model """

    @property
    def root_dir(self) -> str:
        """ Get root directory of containing package """
        return self.__ROOT_DIR

    def _simulation_stopped(self):
        """ Procedures to be performed when simulation has stopped running """
        self.notify_observers_simulation_stopped()

    def _completed_step_increment(self):
        """ Procedures to be performed when completed incrementing simulation by one step """
        self.notify_observers_step_increment()

    # Register observers

    def register_observer_state_change(self, observer):
        """ Register as observer for changes in model state (e.g. running, extinct, timed-out)

            Arguments:
                observer -- the SimulationView object to be added to the state change observers list

            Returns:
                None
        """
        if observer not in self._observers_state_change:
            self._observers_state_change.append(observer)

    def register_observer_simulation_stopped(self, observer):
        """ Register as observer for when simulation stops

            Arguments:
                observer -- the SimulationView object to be added to the simulation stopped observers list

            Returns:
                None
        """
        if observer not in self._observers_simulation_stopped:
            self._observers_simulation_stopped.append(observer)

    def register_observer_step_increment(self, observer):
        """ Register as observer for increment in simulation

            Arguments:
                observer -- the SimulationView object to be added to the step increment observers list

            Returns:
                None
        """
        if observer not in self._observers_step_increment:
            self._observers_step_increment.append(observer)

    # Remove observers

    def remove_observer_state_change(self, observer):
        """ Remove observer from list of observers of changes in model state (e.g. running, extinct, timed-out)

            Arguments:
                observer -- the SimulationView object to be removed from the state change observers list

            Returns:
                None
        """
        try:
            self._observers_state_change.remove(observer)
        except ValueError:
            pass

    def remove_observer_simulation_stopped(self, observer):
        """ Remove observer from list of observers of when simulation has stopped

            Arguments:
                observer -- the SimulationView object to be removed from the simulation stopped observers list

            Returns:
                None
        """
        try:
            self._observers_simulation_stopped.remove(observer)
        except ValueError:
            pass

    def remove_observer_step_increment(self, observer):
        """Remove observer from list of observers of increment in simulation

            Arguments:
                observer -- the SimulationView object to be removed from the step increment observers list

            Returns:
                None
        """
        try:
            self._observers_step_increment.remove(observer)
        except ValueError:
            pass

    # Notify Observers

    def notify_observers_state_change(self, modifier=None):
        """ Notify observers about changes in model state (e.g. running, extinct, timed-out)

            Arguments:
                modifier (object): object that caused state change - to be ignored

            Returns:
                None
        """
        for observer in self._observers_state_change:
            if observer != modifier:
                observer.model_state_change(self)

    def notify_observers_simulation_stopped(self, modifier=None):
        """ Notify observers about when simulation has stopped

            Arguments:
                modifier (object): object that caused simulation to stop - to be ignored

            Returns:
                None
        """
        for observer in self._observers_simulation_stopped:
            if observer != modifier:
                observer.model_simulation_stopped(self)

    def notify_observers_step_increment(self, modifier=None):
        """ Notify observers about increment in simulation

            Arguments:
                modifier (object): object that caused step increment - to be ignored

            Returns:
                None
        """
        for observer in self._observers_step_increment:
            if observer != modifier:
                observer.model_step_increment(self)


class BranchingProcessModel(SimulationModel):
    """
    An abstract base class used to represent all branching process simulation models.

    Attributes
    ----------
        state (BranchingProcessModel): The current state of the simulation
        network (Network): The network that stores the model data

    Methods
    -------
        run_simulation(self, max_time: int, infection_threshold: int) -> None:
            runs the simulation

    """

    def __init__(self):
        # Call superclass constructor
        super().__init__()

        # Set network
        self.network = Network()

        # Set observer lists
        self.observers_graph_change = []

        # State
        self._state = ReadyState(self)

    @property
    def state(self) -> BranchingProcessState:
        """ Get the current state of the simulation """
        return self._state

    @abstractmethod
    def run_simulation(self, max_time: int, infection_threshold: int) -> None:
        """
        Run the simulation until it stops (e.g times out, too many infectious nodes or goes extinct)

            Parameters:
                max_time (int): The maximum number of iterations (eg. days) to be run (simulation stops if reached)
                infection_threshold (int): The maximum number of infectious nodes (simulation stops if reached)

            Returns:
                None
        """

    def graph_changed(self):
        """ Procedures to be performed when the network/graph has changed """
        self.notify_observers_graph_change()

    # Register observers

    def register_observer_graph_change(self, observer):
        """ Register as observer for changes in model graph

            Arguments:
                observer -- the SimulationView object to be added to the state change observers list

            Returns:
                None
        """
        if observer not in self.observers_graph_change:
            self.observers_graph_change.append(observer)

    # Remove observers
    def remove_observer_graph_change(self, observer):
        """ Remove as observer for graph changes

            Arguments:
                observer -- the SimulationView object to be removed from the state change observers list

            Returns:
                None
        """
        try:
            self.observers_graph_change.remove(observer)
        except ValueError:
            pass

    # Notify Observers
    def notify_observers_graph_change(self, modifier=None):
        """ Notify observers about changes in graph

            Arguments:
                modifier (object): object that caused graph change - to be ignored

            Returns:
                None
        """
        for observer in self.observers_graph_change:
            if observer != modifier:
                observer.graph_change(self)
