from typing import List, Optional
from household_contact_tracing.views.branching_process_view import BranchingProcessView
from household_contact_tracing.views.growth_rate_view import GrowthRateView
from household_contact_tracing.branching_process_model import BranchingProcessModel
from household_contact_tracing.views.shell_view import ShellView
from household_contact_tracing.views.csv_file_view import CSVFileView
from household_contact_tracing.views.graph_view import GraphView
from household_contact_tracing.views.graph_pyvis_view import GraphPyvisView
from household_contact_tracing.views.timeline_graph_view import TimelineGraphView
from household_contact_tracing.views.growth_rate_view import GrowthRateView


class BranchingProcessController:
    """
        Branching Process Simulation Controller (MVC pattern)
        Sub-class if a need for different branching process controllers arises.

        Attributes
        ----------
            model (BranchingProcessModel): the model who's state is being recorded
            graph_view (GraphView): matplotlib/pyplot  style simple graph view
            graph_pyvis_view (GraphPyvisView): Pyvis style graph view (outputs to HTML file)
            timeline_view (TimelineGraphView): List of timeline graphs (matplotlib pyplot)
            shell_view (ShellView): console output for any simulation model type
            csv_view (CSVView): csv file output, each row is the final model status information for a simulation run.

        Methods
        -------

            set_graphic_displays(self, display: bool)
                choose whether to show the graphical outputs

            run_simulation(self, max_time: int = 20, max_active_infections: int = 5000)
                runs the simulation

    """

    def __init__(self, model: BranchingProcessModel, additional_views: Optional[List[BranchingProcessView]] = []):
        """
        Constructor for BranchingProcessController

            Parameters:
                model (BranchingProcessModel): The branching process model used to store graph and perform simulation

            Returns:
                new BranchingProcessController
        """
        self._model = model
        self.graph_view = GraphView(model)
        self.graph_pyvis_view = GraphPyvisView(model)
        self.timeline_view = TimelineGraphView(model)
        self.shell_view = ShellView(model)
        self.csv_view = CSVFileView(model)
        self.growth_rate_view = GrowthRateView(model)

        # initialise any views that are required, but included as defaults
        for view in additional_views:
            
            initialised_view = view(model)
            setattr(self, initialised_view.view_name, initialised_view)

        self.set_graphic_displays(False)

    @property
    def model(self) -> BranchingProcessModel:
        """ Get the branching process model used to store graph and perform simulation """
        return self._model

    @model.setter
    def model(self, model: BranchingProcessModel):
        """ Set the branching process model used to store graph and perform simulation """

        # Copy observers across, to new model
        model.copy_observers(self._model)

        # Set new model
        self._model = model

    def set_graphic_displays(self, display: bool):
        """
        Turn on or off all graphical output.

            Parameters:
                display (bool): True to switch graphic displays on, False to switch off.

            Returns:
                None
        """
        self.graph_view.set_display(display)
        self.graph_pyvis_view.set_display(display)
        self.timeline_view.set_display(display)

    def run_simulation(self, state_criteria: dict):
        """
        Run the simulation until it stops (e.g times out, too many infectious nodes or goes extinct)

            Parameters:
                state_criteria: Named variables which are evaluated each step of the model to determine
                  whether the state of the model will change.

            Returns:
                None
        """
        self._model.run_simulation(state_criteria)
