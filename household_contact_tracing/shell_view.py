from household_contact_tracing.simulation_view_interface import SimulationViewInterface


class ShellView(SimulationViewInterface):
    """
        Shell View (for now I just print out everything that I'm registered to observe)
    """

    def __init__(self, controller, model):
        # Viewers can own copies of controller and model (MVC pattern)
        # ... but controller not required yet (no input collected from view)
        #self.controller = controller
        self.model = model

        # Register as observer
        model.register_observer_graph_change(self)
        model.register_observer_param_change(self)
        model.register_observer_state_change(self)
        model.register_observer_step_increment(self)

    def model_param_change(self, subject):
        """ Respond to parameter change(s) """
        print('shell view observed param change')

    def graph_change(self, subject):
        """ Respond to changes in graph (nodes/households network) """
        print('shell view observed graph change')

    def model_state_change(self, subject):
        """ Respond to changes in model state (e.g. running, extinct, timed-out) """
        print('shell view observed state change. New state: {}'.format(subject.state.name))

    def model_step_increment(self, subject):
        """ Respond to increment in simulation """
        print('shell view observed that Model has been incremented by one step')
