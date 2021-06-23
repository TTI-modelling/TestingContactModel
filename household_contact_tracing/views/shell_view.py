from household_contact_tracing.views.simulation_view import SimulationView
from household_contact_tracing.simulation_model import BranchingProcessModel


class ShellView(SimulationView):
    """
        Shell View (for now I just print out everything that I'm registered to observe)
    """

    def __init__(self, model: BranchingProcessModel):
        # Viewers can also own copies of controller and model (MVC pattern)
        # ... but controller not required yet (no input collected from view)
        #self.controller = controller
        self._model = model

        # Register default observers
        self._model.register_observer_state_change(self)
        self._model.register_observer_simulation_stopped(self)

    def set_display(self, show: bool):
        if show:
            self._model.register_observer_graph_change(self)
            self._model.register_observer_state_change(self)
            self._model.register_observer_step_increment(self)
            self._model.register_observer_simulation_stopped(self)
        else:
            self._model.remove_observer_graph_change(self)
            self._model.remove_observer_state_change(self)
            self._model.remove_observer_step_increment(self)
            self._model.remove_observer_simulation_stopped(self)

    def graph_change(self, subject: BranchingProcessModel):
        """ Respond to changes in graph (nodes/households network) """
        print('Graph changed')

    def model_state_change(self, subject: BranchingProcessModel):
        """ Respond to changes in model state (e.g. running, extinct, timed-out) """
        print('State change: New state: {}'.format(subject.state))

    def model_step_increment(self, subject: BranchingProcessModel):
        """ Respond to increment in simulation """
        print('Model has been incremented by one step')

    def model_simulation_stopped(self, subject: BranchingProcessModel):
        print('Simulation has stopped running')