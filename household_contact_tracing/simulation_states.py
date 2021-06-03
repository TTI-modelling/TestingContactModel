from abc import ABC, abstractmethod


class SimulationState(ABC):

    """
        Simulation State Abstract class (State pattern)
    """

    def __init__(self, simulation_model):
        self._simulation_model = simulation_model

    def get_state_name(self: str):
        """ Return the state name."""
        return self.__class__.__name__

    @property
    def name(self):
        return self.get_state_name()

    @abstractmethod
    def start_run(self):
        """ The simulation has started running."""
        pass

    @abstractmethod
    def timed_out(self):
        """ The simulation timed out """
        pass

    @abstractmethod
    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        pass

    @abstractmethod
    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        pass

    @abstractmethod
    def initialise(self):
        """ The simulation was re-set to initialised """
        pass


class ReadyState(SimulationState):
    """
        Simulation is ready to start
    """

    def initialise(self):
        """ The simulation was re-set to initial pre-run state """
        self._simulation_model.state = self._simulation_model.ready_state

    def start_run(self):
        """ The simulation has started running."""
        self._simulation_model.state = self._simulation_model.running_state
        self._simulation_model.notify_observers_state_change()

    def timed_out(self):
        """ The simulation timed out """
        raise ValueError('Not possible to time-out during ready state!')

    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        raise ValueError('Not possible to go extinct during ready state!')

    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        raise ValueError('Not possible to reach max num infectious nodes during ready state!')


class RunningState(SimulationState):
    """
        Simulation is running
    """

    def initialise(self):
        """ The simulation was re-set to initial pre-run state """
        raise ValueError('Cannot initialise whilst running. Wait for stop and then re-initialise model')

    def start_run(self):
        """ The simulation has started running."""
        raise ValueError('Already in Running state, cannot start run')

    def timed_out(self):
        """ The simulation timed out """
        self._simulation_model.state = self._simulation_model.timed_out_state
        self._simulation_model.notify_observers_state_change()

    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        self._simulation_model.state = self._simulation_model.extinct_state
        self._simulation_model.notify_observers_state_change()

    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        self._simulation_model.state = self._simulation_model.max_nodes_infectious_state
        self._simulation_model.notify_observers_state_change()


class ExtinctState(SimulationState):
    """
        Simulated outbreak has gone extinct
    """

    def initialise(self):
        """ The simulation was re-set to initial pre-run state """
        raise ValueError('Not possible to initialise from extinct state. Re-initialise the model')

    def start_run(self):
        """ The simulation has started running."""
        self._simulation_model.state = self._simulation_model.running_state
        self._simulation_model.notify_observers_state_change()

    def timed_out(self):
        """ The simulation timed out """
        raise ValueError('Can not time out if extinct!')

    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        raise ValueError('Cannot go extinct if already extinct')

    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        raise ValueError('Can not reach max infectious nodes if extinct')


class TimedOutState(SimulationState):
    """
        Simulated outbreak has timed out
    """

    def initialise(self):
        """ The simulation was re-set to initial pre-run state """
        raise ValueError('Can not initialise from timed out state. Re-initialise the model')

    def start_run(self):
        """ The simulation has started running."""
        self._simulation_model.state = self._simulation_model.running_state
        self._simulation_model.notify_observers_state_change()

    def timed_out(self):
        """ The simulation timed out """
        raise ValueError('Can not time out if already timed out.')

    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        raise ValueError('Can not go extinct if timed out.')

    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        raise ValueError('Can not reach max infectious nodes if timed out.')


class MaxNodesInfectiousState(SimulationState):
    """
        Simulated outbreak has reached maximum number of infectious nodes
    """

    def initialise(self):
        """ The simulation was re-set to initial pre-run state """
        raise ValueError('Can not initialise from max nodes reached state. Re-initialise the model')

    def start_run(self):
        """ The simulation has started running."""
        self._simulation_model.state = self._simulation_model.running_state
        self._simulation_model.notify_observers_state_change()

    def timed_out(self):
        """ The simulation timed out """
        raise ValueError('Can not time out if already reached max nodes infectious.')

    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        raise ValueError('Can not go extinct if reached max nodes infectious.')

    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        raise ValueError('Can not reach maximum number of infectious nodes if it is already reached.')