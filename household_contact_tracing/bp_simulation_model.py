from household_contact_tracing.simulation_model_interface import SimulationModelInterface
from household_contact_tracing.simulation_view_interface import SimulationViewInterface
from household_contact_tracing.simulation_states import SimulationStateInterface, ReadyState, RunningState, \
    ExtinctState, TimedOutState, MaxNodesInfectiousState


class BPSimulationModel(SimulationModelInterface):
    """
        Branching Process Simulation Controller
    """

    def __init__(self):
        # Set observer lists
        self._observers_param_change = []
        self._observers_graph_change = []
        self._observers_state_change = []
        self._observers_step_increment = []

        # States
        self._ready_state = ReadyState(self)
        self._running_state = RunningState(self)
        self._extinct_state = ExtinctState(self)
        self._timed_out_state = TimedOutState(self)
        self._max_nodes_infectious_state = MaxNodesInfectiousState(self)
        self._state = self._ready_state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: SimulationStateInterface):
        self._state = state

    @property
    def ready_state(self):
        return self._ready_state

    @property
    def running_state(self):
        return self._running_state

    @property
    def extinct_state(self):
        return self._extinct_state

    @property
    def timed_out_state(self):
        return self._timed_out_state

    @property
    def max_nodes_infectious_state(self):
        return self._max_nodes_infectious_state


    ### Fulfill inherited interface methods: ###

    def initialised_simulation(self):
        """ Initialise/Reset the simulation to starting values."""
        # NOTIFY OF STATE CHANGE
        print('Initialised simulation, so set simulator state to Ready')
        self._state.reset()

    def updated_parameters(self):
        """ Increment simulation by one step """
        self.notify_observer_param_change()

    def started_simulation(self,  num_steps: int, infection_threshold: int = 5000):
        """ Start the simulation running."""
        # NOTIFY OF STATE CHANGE
        print('Started simulation, so set simulator state to Running')
        self._state.start_run()

    def graph_changed(self):
        """ The graph has changed """
        self.notify_observer_graph_change()

    def completed_step_increment(self):
        """ Completed incrementing simulation by one step """
        self.notify_observer_step_increment()

    # Register observers

    def register_observer_param_change(self, observer: object):
        """ Register as observer for parameter changes

        Arguments:
            observer -- the object to be added to the param change observers list
        """
        if observer not in self._observers_param_change:
            self._observers_param_change.append(observer)

    def register_observer_graph_change(self, observer: SimulationViewInterface):
        """ Register as observer for changes in model (nodes/households network)

        Arguments:
            observer -- the object to be added to the model change observers list
        """
        if observer not in self._observers_graph_change:
            self._observers_graph_change.append(observer)

    def register_observer_state_change(self, observer: SimulationViewInterface):
        """ Register as observer for changes in model state (e.g. running, extinct, timed-out)
        Arguments:
            observer -- the object to be added to the state change observers list
        """
        if observer not in self._observers_state_change:
            self._observers_state_change.append(observer)

    def register_observer_step_increment(self, observer: SimulationViewInterface):
        """ Register as observer for increment in simulation
        Arguments:
            observer -- the object to be added to the increment observers list
        """
        if observer not in self._observers_step_increment:
            self._observers_step_increment.append(observer)

    # Remove observers
    def remove_observer_param_change(self, observer: SimulationViewInterface):
        """ Remove as observer for parameter changes """
        try:
            self._observers_param_change.remove(observer)
        except ValueError:
            pass

    def remove_observer_model_change(self, observer: SimulationViewInterface):
        """ Remove as observer for changes in model (nodes/households network) """
        try:
            self._observers_model_change.remove(observer)
        except ValueError:
            pass

    def remove_observer_state_change(self, observer: SimulationViewInterface):
        """ Remove as observer for changes in model state (e.g. running, extinct, timed-out) """
        try:
            self._observers_state_change.remove(observer)
        except ValueError:
            pass

    def remove_observer_step_increment(self, observer: SimulationViewInterface):
        """ Remove as observer for increment in simulation """
        try:
            self._observers_step_increment.remove(observer)
        except ValueError:
            pass

    # Notify Observers
    def notify_observer_param_change(self, modifier=None):
        """ Notify observer about parameter changes """
        for observer in self._observers_param_change:
            if observer != modifier:
                observer.update_model_param_change(self)

    def notify_observer_graph_change(self, modifier=None):
        """ Notify observer about changes in graph (nodes/households network) """
        for observer in self._observers_graph_change:
            if observer != modifier:
                observer.update_graph_change(self)

    def notify_observer_state_change(self, modifier=None):
        """ Notify observer about changes in model state (e.g. running, extinct, timed-out)  """
        for observer in self._observers_state_change:
            if observer != modifier:
                observer.update_model_state_change(self)

    def notify_observer_step_increment(self, modifier=None):
        """ Notify observer about  increment in simulation """
        for observer in self._observers_step_increment:
            if observer != modifier:
                observer.update_model_step_increment(self)