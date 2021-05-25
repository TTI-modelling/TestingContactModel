from household_contact_tracing.simulation_model import SimulationModel
from household_contact_tracing.views.shell_view import ShellView
from household_contact_tracing.views.graph_view import GraphView
from household_contact_tracing.views.graph_pyvis_view import GraphPyvisView
from household_contact_tracing.views.timeline_graph_view import TimelineGraphView


class SimulationController:
    """ Branching Process Simulation Controller (MVC pattern)
        Consider sub-classing and making this class abstract, if a need for different controllers
        arises.
    """

    def __init__(self, model: SimulationModel):
        self._model = model
        self.shell_view = ShellView(self, model)
        self.graph_view = GraphView(self, model)
        self.graph_pyvis_view = GraphPyvisView(self, model)
        self.timeline_view = TimelineGraphView(self, model)

    @property
    def model(self) -> SimulationModel:
        return self._model

    @model.setter
    def model(self, model: SimulationModel):
        self._model = model

    def run_simulation(self, max_time: int = 20, infection_threshold: int = 5000):
        """ Run the simulation."""
        self._model.run_simulation(max_time, infection_threshold)
