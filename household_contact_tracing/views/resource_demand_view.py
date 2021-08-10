"""
Contains a view that produces outputs on the different resources consumed by the model.
This could be the number of required tests, the number of positive tests, the amount of isolation days.
"""
from numpy import append
from household_contact_tracing.branching_process_model import BranchingProcessModel
from household_contact_tracing.views.branching_process_view import BranchingProcessView
from household_contact_tracing.network import TestType
import pandas as pd

class ResourceDemandView:

    def __init__(self, model: BranchingProcessModel):
        self._model = model
        self.show = False

        self._model.register_observer_step_increment(self)

        # create empty lst attributes for storing data
        self.positive_PCR_counts = []

    def model_state_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in model state (e.g. running, extinct, timed-out)

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    def model_step_increment(self, subject: BranchingProcessModel):
        """
        Respond to single step increment in simulation

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        self._model = subject
        
        self.record_resource_demand()

    def model_simulation_stopped(self, subject: BranchingProcessModel):
        """
        Respond to end of simulation run

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        # This view does not need to do anything upon a simulation stopping
        pass

    def graph_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in graph (nodes/households network)

            Parameters:
                subject (SimulationModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        # This view does not need to do anything upon a graph change.
        pass

    def set_display(self, show: bool):
        """
        Sets whether this view is displayed or not.

            Parameters:
                show (bool): To display this view, set to True

            Returns:
                None
        """
        self.show = show

    def record_resource_demand(self):
        """Inspect the current state of the model, compute the resource demand, and store.
        """
        self.positive_PCR_counts.append(self.get_positive_PCR_count())

    def get_positive_PCR_count(self) -> int:
        """Inspects a model, and compute the number of positive tests that will occur that timestep.
        """

        # one is subtracted from the time, since +1 gets added to the time at the end of a simulation day.
        # by subtracting, we get the quantity during the last simulated day
        return len([
            node
            for node 
            in self._model.network.all_nodes()
            if node.lfd_testing.positive_test_time == self._model.time - 1
            and node.lfd_testing.avenue_of_testing == TestType.pcr 
        ])

    def resource_demand_df(self):
        """Returns the resource demand as a Pandas dataframe.
        """
        return pd.DataFrame({
            'time': list(range(self._model.time)),
            'positive_PCR_counts': self.positive_PCR_counts
        })
