class SimulationControllerInterface:
    '''
        Simulation Controller interface (MVC pattern)
    '''
    def reset(self):
        """ Reset the simulation model."""
        pass

    def run_simulation(self):
        """ Run the simulation."""
        pass

    def set_params(self, params):
        """ Update parameters for the simulation."""
        pass
