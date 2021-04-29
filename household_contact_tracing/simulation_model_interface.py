class SimulationModelInterface:
    '''
        Simulation Model interface (MVC pattern)
    '''

    def simulation_initialised(self):
        """ Initialise/Reset the simulation to starting values."""
        pass

    def updated_parameters(self, params):
        """ Set new params for simulation """
        pass

    def simulation_started(self,  num_steps: int, infection_threshold: int = 5000):
        """ Start the simulation running."""
        pass

    def completed_step_increment(self):
        """ Completed incrementing simulation by one step """
        pass

    # Register / Remove obervers

    def register_observer_param_change(self, observer: object):
        """ Register as observer for parameter changes """
        pass

    def register_observer_graph_change(self, observer: object):
        """ Register as observer for changes in graph (nodes/households network) """
        pass

    def register_observer_state_change(self, observer: object):
        """ Register as observer for changes in model state (e.g. running, extinct, timed-out) """
        pass

    def register_observer_run_stopped(self, observer: object):
        """ Register as observer for when simulation run has stopped """
        pass

    def register_observer_increment(self, observer: object):
        """ Register as observer for increment in simulation """
        pass

    def remove_observer_param_change(self, observer: object):
        """ Remove as observer for parameter changes """
        pass

    def remove_observer_graph_change(self, observer: object):
        """ Remove as observer for changes in graph (nodes/households network) """
        pass

    def remove_observer_state_change(self, observer: object):
        """ Remove as observer for changes in model state (e.g. running, extinct, timed-out) """
        pass

    def remove_observer_run_stopped(self, observer: object):
        """ Remove observer for when simulation run has stopped """
        pass

    def remove_observer_increment(self, observer: object):
        """ Remove as observer for increment in simulation """
        pass

    # Notify Observers
    def notify_observers_param_change(self, modifier=None):
        """ Notify observers about parameter changes """
        pass

    def notify_observers_graph_change(self, modifier=None):
        """ Notify observers about changes in graph (nodes/households network) """
        pass

    def notify_observers_state_change(self, modifier=None):
        """ Notify observer about changes in model state (e.g. running, extinct, timed-out)  """
        pass

    def notify_observers_simulation_stopped(self, modifier=None):
        """ Notify observer about when simulation run has stopped """
        pass

    def notify_observers_step_increment(self, modifier=None):
        """ Notify observer about  increment in simulation """
        pass

