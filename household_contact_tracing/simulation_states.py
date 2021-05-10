class SimulationStateInterface:
    """
        Simulation State interface (State pattern)
    """

    def start_run(self):
        """ The simulation has started running."""
        pass

    def timed_out(self):
        """ The simulation timed out """
        pass

    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        pass

    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        pass

    def initialise(self):
        """ The simulation was re-set to initialised """
        pass


class SimulationState(SimulationStateInterface):
    def __init__(self, simulation_model):
        self._simulation_model = simulation_model

    def get_state_name(self: str):
        """ Return the state name."""
        return self.__class__.__name__

    @property
    def name(self):
        return self.get_state_name()


class ReadyState(SimulationState):
    """
        Simulation is ready to start
    """

    def initialise(self):
        """ The simulation was re-set to initial pre-run state """
        print('In ready/initialised state!')
        self._simulation_model.state = self._simulation_model.ready_state

    def start_run(self):
        """ The simulation has started running."""
        print('Changing to running state')
        self._simulation_model.state = self._simulation_model.running_state
        self._simulation_model.notify_observers_state_change()

    def timed_out(self):
        """ The simulation timed out """
        print('You can\'t time-out during ready state!')

    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        print('You can\'t go extinct during ready state!')

    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        print('You can\'t reach max num infectious nodes during ready state!')


class RunningState(SimulationState):
    """
        Simulation is running
    """

    def initialise(self):
        """ The simulation was re-set to initial pre-run state """
        print('Cannot initialise whilst running. Wait for stop and then re-initialise model')

    def start_run(self):
        """ The simulation has started running."""
        print('Already in Running state')

    def timed_out(self):
        """ The simulation timed out """
        print('Timed out state')
        self._simulation_model.state = self._simulation_model.timed_out_state
        self._simulation_model.notify_observers_state_change()

    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        print('Gone extinct state')
        self._simulation_model.state = self._simulation_model.extinct_state
        self._simulation_model.notify_observers_state_change()

    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        print('Max infectious nodes threshold met')
        self._simulation_model.state = self._simulation_model.max_nodes_infectious_state
        self._simulation_model.notify_observers_state_change()


class ExtinctState(SimulationState):
    """
        Simulated outbreak has gone extinct
    """

    def initialise(self):
        """ The simulation was re-set to initial pre-run state """
        print('Can\'t initialise from extinct state. Re-initialise model')

    def start_run(self):
        """ The simulation has started running."""
        print('Can\'t start run from extinct state. Re-initialise model')

    def timed_out(self):
        """ The simulation timed out """
        print('Cannot time out if extinct!')

    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        print('Already extinct!')

    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        print('Cannot reach max infectious nodes if extinct!')


class TimedOutState(SimulationState):
    """
        Simulated outbreak has timed out
    """

    def initialise(self):
        """ The simulation was re-set to initial pre-run state """
        print('Can\'t initialise from timed out state. Re-initialise model')

    def start_run(self):
        """ The simulation has started running."""
        print('Can\'t start run from timed-out state. Re-initialise model')

    def timed_out(self):
        """ The simulation timed out """
        print('Already timed out!')

    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        print('Cannot go extinct if timed out!')

    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        print('Cannot reach max infectious nodes if timed out!')


class MaxNodesInfectiousState(SimulationState):
    """
        Simulated outbreak has reached maximum number of infectious nodes
    """

    def initialise(self):
        """ The simulation was re-set to initial pre-run state """
        print('Can\'t initialise from max nodes reached state. Re-initialise model')

    def start_run(self):
        """ The simulation has started running."""
        print('Can\'t start run from max infected nodes state. Re-initialise model')

    def timed_out(self):
        """ The simulation timed out """
        print('Cannot time out if already reached max nodes infectious!')

    def go_extinct(self):
        """ The simulation went extinct (no more infectious nodes) """
        print('Cannot go extinct if reached max nodes infectious!')

    def max_nodes_infectious(self):
        """  The max number of infectious nodes was reached """
        print('Already reach max infectious nodes!')