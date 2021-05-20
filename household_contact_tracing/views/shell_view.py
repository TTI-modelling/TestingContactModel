from household_contact_tracing.views.simulation_view import SimulationView


class ShellView(SimulationView):
    """
        Shell View (for now I just print out everything that I'm registered to observe)
    """

    def __init__(self, controller, model):
        # Viewers can own copies of controller and model (MVC pattern)
        # ... but controller not required yet (no input collected from view)
        #self.controller = controller
        self._model = model

        # Register as observer
        self._model.register_observer_graph_change(self)
        self._model.register_observer_param_change(self)
        self._model.register_observer_state_change(self)
        self._model.register_observer_step_increment(self)
        self._model.register_observer_simulation_stopped(self)

    def set_display(self, show: bool):
        if show:
            self._model.register_observer_graph_change(self)
            self._model.register_observer_param_change(self)
            self._model.register_observer_state_change(self)
            self._model.register_observer_step_increment(self)
            self._model.register_observer_simulation_stopped(self)
        else:
            self._model.remove_observer_graph_change(self)
            self._model.remove_observer_param_change(self)
            self._model.remove_observer_state_change(self)
            self._model.remove_observer_step_increment(self)
            self._model.remove_observer_simulation_stopped(self)

    def model_param_change(self, subject):
        """ Respond to parameter change(s) """
        print('shell view observed param change')

    def graph_change(self, subject):
        """ Respond to changes in graph (nodes/households network) """
        #print('shell view observed graph change')

    def model_state_change(self, subject):
        """ Respond to changes in model state (e.g. running, extinct, timed-out) """
        print('shell view observed state change. New state: {}'.format(subject.state.name))

    def model_step_increment(self, subject):
        """ Respond to increment in simulation """
        #print('shell view observed that Model has been incremented by one step')

    def model_simulation_stopped(self, subject):
        print('shell view observed that simulation has stopped running')
