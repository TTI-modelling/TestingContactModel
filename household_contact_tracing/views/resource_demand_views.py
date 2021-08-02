'''
Contains several views that can be used to record the demand for various resources
at each time step of a simulation. Resources might include the total number of individuals
attempting to book a test at each timestep
'''
from household_contact_tracing.views.branching_process_view import BranchingProcessView
from household_contact_tracing.branching_process_models import BranchingProcessModel

class PositiveTestRecord(BranchingProcessView):
    """This view records the number of individuals at each time step who have tested postivie for 
    for the first time during that timestep.
    """

    def __init__(self, model: BranchingProcessModel):
        
        self._model = model
        self.view_name = 'positive_test_record'

        self.positive_tests_requested = []

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
        Sets whether this view is displayed or not.

            Parameters:
                show (bool): To display this view, set to True

            Returns:
                None
        """
        

    def positive_tests_over_time(self):
        """Returns a list of the number of positive tests received at each timestep.
        """

        positive_test_times = [
            node.positive_test_time 
            for node in self._model.network.all_nodes() 
            if node.positive_test_time is not None
        ]

        return [
            positive_test_times.count(time)
            for time in range(self._model.time)
        ]

    def active_infections(self):
        pass