from household_contact_tracing.simulation_model import SimulationModel
from household_contact_tracing.shell_view import ShellView
from household_contact_tracing.graph_view import GraphView


class SimulationController:
    """ Branching Process Simulation Controller (MVC pattern)
        Consider sub-classing and making this class abstract, if a need for different controllers arises.
    """


    def __init__(self, model: SimulationModel):
        self._model = model
        self.shellView = ShellView(self, model)
        self.graphView = GraphView(self, model)

    @property
    def model(self) -> SimulationModel:
        return self._model

    @model.setter
    def model(self, model: SimulationModel):
        self._model = model

    def set_show_all_graphs(self, show_all):
        self.graphView.set_show_all_graphs(show_all)

    def run_simulation(self, num_steps: int = 20, infection_threshold: int = 5000):
        """ Run the simulation."""
        self._model.run_simulation(num_steps, infection_threshold)

    def set_params(self, params):
        """ Set parameters for the simulation."""
        self._model.set_params(params)
