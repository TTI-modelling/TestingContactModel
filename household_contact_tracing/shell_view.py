from household_contact_tracing.simulation_view_interface import SimulationViewInterface


class ShellView(SimulationViewInterface):
    """
        Shell View
    """

    def __init__(self, controller, model):
        # Viewers own copies of controller and model (MVC pattern)
        self.controller = controller
        self.model = model

        # Register as observer
        model.register_observer_model_change(self)
        model.register_observer_param_change(self)
        model.register_observer_state_change(self)
        model.register_observer_step_increment(self)

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
        print('param change to [need to implement which one here!]')

    def update_model_change(self, subject):
        """ Respond to changes in model (nodes/households network) """
        print('model change to [need to implement showing change here!]')

    def update_model_state_change(self, subject):
        """ Respond to changes in model state (e.g. running, extinct, timed-out) """
        print('state change to [need to implement which one here!]')

    def update_model_step_increment(self, subject):
        """ Respond to increment in simulation """
        print('Model has been incremented by one step')
