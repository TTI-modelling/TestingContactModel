from household_contact_tracing.simulation_view_interface import SimulationViewInterface


class GraphView(SimulationViewInterface):
    '''
        Graph View
    '''

    def __init__(self, controller, model):
        # Viewers own copies of controller and model (MVC pattern)
        self.controller = controller
        self.model = model

        # Register as observer
        model.register_observer_model_change(self)

    def update_model_change(self, subject):
        """ Respond to changes in model (nodes/households network) """
        print('model change to [need to implement showing change here!]')

