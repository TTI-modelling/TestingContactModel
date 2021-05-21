from household_contact_tracing.simulation_model import SimulationModel
from household_contact_tracing.views.shell_view import ShellView
from household_contact_tracing.views.graph_view import GraphView
from household_contact_tracing.views.timeline_graph_view import TimelineGraphView


class SimulationController:
    """ Branching Process Simulation Controller (MVC pattern)
        Consider sub-classing and making this class abstract, if a need for different controllers
        arises.
    """

    def __init__(self, model: SimulationModel):
        self._model = model
        self.shellView = ShellView(self, model)
        self.graphView = GraphView(self, model)
        self.timelineView = TimelineGraphView(self, model)

    @property
    def model(self) -> SimulationModel:
        return self._model

    @model.setter
    def model(self, model: SimulationModel):
        self._model = model

    def set_show_all_graphs(self, show_all: bool):
        self.graphView.set_show_all_graphs(show_all)

    def set_show_graphs(self, show: bool):
        self.graphView.set_display(show)

    def set_shell_output(self, show: bool):
        self.shellView.set_display(show)

    def set_timeline_view(self, show: bool):
        self.timelineView.set_display(show)

    def run_simulation(self, max_time: int = 20, infection_threshold: int = 5000):
        """ Run the simulation."""
        self._model.run_simulation(max_time, infection_threshold)
