class SimulationModelInterface:
    '''
        Simulation Controller interface (MVC pattern)
    '''

    def initialise_simulation(self):
        """ Initialise/Reset the simulation to starting values."""
        pass

    def update_parameters(self, params):
        """ Set new params for simulation """
        pass

    def start_simulation(self,  num_steps: int, infection_threshold: int = 5000):
        """ Start the simulation running."""
        pass

    def simulate_one_step(self):
        """ Just do a single step of simulation """
        pass

    def completed_step_increment(self):
        """ Completed incrementing simulation by one step """
        pass


    def register_observer_param_change(self, observer: object):
        """ Register as observer for parameter changes """
        pass

    def register_observer_model_change(self, observer: object):
        """ Register as observer for changes in model (nodes/households network) """
        pass

    def register_observer_state_change(self, observer: object):
        """ Register as observer for changes in model state (e.g. running, extinct, timed-out) """
        pass

    def register_observer_increment(self, observer: object):
        """ Register as observer for increment in simulation """
        pass

    def remove_observer_param_change(self, observer: object):
        """ Remove as observer for parameter changes """
        pass

    def remove_observer_model_change(self, observer: object):
        """ Remove as observer for changes in model (nodes/households network) """
        pass

    def remove_observer_state_change(self, observer: object):
        """ Remove as observer for changes in model state (e.g. running, extinct, timed-out) """
        pass

    def remove_observer_increment(self, observer: object):
        """ Remove as observer for increment in simulation """
        pass

    # Notify Observers
    def notify_observer_param_change(self, modifier=None):
        """ Notify observer about parameter changes """
        pass

    def notify_observer_model_change(self, modifier=None):
        """ Notify observer about changes in model (nodes/households network) """
        pass

    def notify_observer_state_change(self, modifier=None):
        """ Notify observer about changes in model state (e.g. running, extinct, timed-out)  """
        pass

    def notify_observer_increment(self, modifier=None):
        """ Notify observer about  increment in simulation """
        pass

