# Code to estimate the growth rate of a simulated branching process

from household_contact_tracing.views.branching_process_view import BranchingProcessView
from household_contact_tracing.branching_process_model import BranchingProcessModel

class GrowthRateView(BranchingProcessView):

    """
    View that estimates the growth rate of a completed simulation
    """

    def __init__(self, model: BranchingProcessModel):
        
        self._model = model

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
        pass

    def model_simulation_stopped(self, subject: BranchingProcessModel):
        """
        Respond to end of simulation run

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    def graph_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in graph (nodes/households network)

            Parameters:
                subject (SimulationModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    def set_display(self, show: bool):
        """
        Sets whether this pyvis graph view is displayed or not.

            Parameters:
                show (bool): To display this view, set to True

            Returns:
                None
        """
        pass

    def get_infection_times(self):
        """
        Returns a list containing the times at which each node was infected
        """
        return [node.time_infected for node in self._model.network.all_nodes()]

    def get_daily_incidence(self):
        """Returns a list of the new infections at each time point.

        The list contains [time, incidence] pairs
        """
        
        infection_times = self.get_infection_times()
        return([
            infection_times.count(t)
            for t in range(self._model.time)
        ])

    def estimate_growth_rate(self):

        print('estimating growth rate')
