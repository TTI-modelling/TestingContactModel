class SimulationViewInterface:
    '''
        Simulation View interface (MVC pattern)
    '''

    def reset(self):
        """ Initialise/Reset the simulation to starting values."""
        pass

    def run(self):
        """ Start the simulation running."""
        pass

    def set_params(self, params):
        """ Set new parameters for the simulation."""
        pass

    def update_model_param_change(self, subject):
        """ Respond to parameter change(s) """
        pass

    def update_model_change(self, subject):
        """ Respond to changes in model (nodes/households network) """
        pass

    def update_model_state_change(self, subject):
        """ Respond to changes in model state (e.g. running, extinct, timed-out) """
        pass

    def update_model_step_increment(self, subject):
        """ Respond to single step increment in simulation """
        pass
