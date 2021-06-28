from abc import ABC, abstractmethod

from household_contact_tracing.branching_process_model import BranchingProcessModel


class BranchingProcessView(ABC):
    """
        Branching process View (Abstract) (MVC pattern)
        This is the abstract parent of all view classes. Sub-class for adding new views
        (displays of branching process model outputs).

        Methods (Abstract)
        ----------
            set_display(self, show: bool)
                sets whether the display is to be shown or not


            MVC related abstract methods. Responses to change events broadcast by the model:

            graph_change(self, subject: BranchingProcessModel)
                Respond to changes in graph (nodes/households network)

            model_state_change(self, subject: BranchingProcessModel)
                Respond to changes in model state (e.g. running, extinct, timed-out)

            model_step_increment(self, subject: BranchingProcessModel)
                Respond to single step increment in simulation

            model_simulation_stopped(self, subject: BranchingProcessModel)
                Respond to simulation stopping
    """

    @abstractmethod
    def set_display(self, show: bool):
        """
        Sets whether this view is displayed or not.

            Parameters:
                show (bool): To display this view, set to True

            Returns:
                None
        """
        pass

    @abstractmethod
    def graph_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in graph (nodes/households network)

            Parameters:
                subject (BranchingProcessModel): The simulation model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    @abstractmethod
    def model_state_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in model state (e.g. running, extinct, timed-out)

            Parameters:
                subject (BranchingProcessModel): The simulation model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    @abstractmethod
    def model_step_increment(self, subject: BranchingProcessModel):
        """
        Respond to single step increment in simulation

            Parameters:
                subject (BranchingProcessModel): The simulation model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    @abstractmethod
    def model_simulation_stopped(self, subject: BranchingProcessModel):
        """
        Respond to simulation stopping

            Parameters:
                subject (BranchingProcessModel): The simulation model being displayed by this simulation view.

            Returns:
                None
        """
        pass
