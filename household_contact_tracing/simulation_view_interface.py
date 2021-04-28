class SimulationViewInterface:
    '''
        Simulation View interface (MVC pattern)
    '''

    def update_model_param_change(self, subject):
        """ Respond to parameter change(s) """
        pass

    def update_graph_change(self, subject):
        """ Respond to changes in graph (nodes/households network) """
        pass

    def update_model_state_change(self, subject):
        """ Respond to changes in model state (e.g. running, extinct, timed-out) """
        pass

    def update_model_step_increment(self, subject):
        """ Respond to single step increment in simulation """
        pass
