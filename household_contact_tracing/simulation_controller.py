from abc import ABC

from household_contact_tracing.simulation_model import BranchingProcessModel
from household_contact_tracing.views.shell_view import ShellView
from household_contact_tracing.views.graph_view import GraphView
from household_contact_tracing.views.graph_pyvis_view import GraphPyvisView
from household_contact_tracing.views.timeline_graph_view import TimelineGraphView


class SimulationController(ABC):
    """
        Simulation Controller (MVC pattern)
        This controller owns both the model and the views, and is the mediator
        Sub-class if a need for different controllers  arises.

        Attributes
        ----------
        model (BranchingProcessModel): the model who's state is being recorded
        shell_view (ShellView): console output for any simulation model type

        Methods
        -------

    """

    def __init__(self, model: BranchingProcessModel):
        self._model = model
        self.shell_view = ShellView(self, model)

    @property
    def model(self) -> BranchingProcessModel:
        return self._model

    @model.setter
    def model(self, model: BranchingProcessModel):
        self._model = model


class BranchingProcessController(SimulationController):
    """
        Branching Process Simulation Controller (MVC pattern)
        Sub-class if a need for different branching process controllers arises.

        Attributes
        ----------
        graph_view (GraphView): matplotlib/pyplot  style simple graph view
        graph_pyvis_view (GraphPyvisView): Pyvis style graph view (outputs to HTML file)
        timeline_view (TimelineGraphView): List of timeline graphs (matplotlib pyplot)

        Methods
        -------

        set_display(self, display: bool)
            choose whether to show the graphical outputs

        run_simulation(self, max_time: int = 20, infection_threshold: int = 5000)
            runs the simulation

    """

    def __init__(self, model: BranchingProcessModel):
        # Call superclass constructor
        super().__init__(model)

        self.graph_view = GraphView(self, model)
        self.graph_pyvis_view = GraphPyvisView(self, model)
        self.timeline_view = TimelineGraphView(self, model)

        self.set_graphic_displays(False)

    def set_graphic_displays(self, display: bool):
        """Turn on or off all graphical output."""
        self.graph_view.set_display(display)
        self.graph_pyvis_view.set_display(display)
        self.timeline_view.set_display(display)

    def run_simulation(self, max_time: int = 20, infection_threshold: int = 5000):
        """ Run the simulation."""
        self._model.run_simulation(max_time, infection_threshold)
