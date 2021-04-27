from household_contact_tracing.simulation_model_interface import SimulationModelInterface
from household_contact_tracing.simulation_view_interface import SimulationViewInterface

class BPSimulationModel(SimulationModelInterface):
    """
        Branching Process Simulation Controller
    """

    def __init__(self):
        # Set observer lists
        self._observers_param_change = []
        self._observers_model_change = []
        self._observers_state_change = []
        self._observers_step_increment = []

    ### Fulfill inherited interface methods: ###

    def initialise_simulation(self):
        """ Initialise/Reset the simulation to starting values."""
        # NOTIFY OF STATE CHANGE HERE (ONCE SIMULATION STATES ARE FULLY IMPLEMENTED)
        pass

    def update_parameters(self):
        """ Increment simulation by one step """
        self.notify_observer_param_change()

    def start_simulation(self,  num_steps: int, infection_threshold: int = 5000):
        """ Start the simulation running."""
        # NOTIFY OF STATE CHANGE HERE (ONCE SIMULATION STATES ARE FULLY IMPLEMENTED)
        pass

    def simulate_one_step(self):
        pass

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

    def register_observer_model_change(self, observer: SimulationViewInterface):
        """ Register as observer for changes in model (nodes/households network)

        Arguments:
            observer -- the object to be added to the model change observers list
        """
        if observer not in self._observers_model_change:
            self._observers_model_change.append(observer)

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

    def notify_observer_model_change(self, modifier=None):
        """ Notify observer about changes in model (nodes/households network) """
        for observer in self._observers_model_change:
            if observer != modifier:
                observer.update_model_change(self)

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
