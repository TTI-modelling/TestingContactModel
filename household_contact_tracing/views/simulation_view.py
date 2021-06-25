from abc import ABC, abstractmethod

from household_contact_tracing.simulation_model import SimulationModel


class SimulationView(ABC):
    """
        Simulation View (Abstract) (MVC pattern)
        This is the abstract parent of all view classes. Sub-class for adding new views
        (displays of simulation outputs).

        Methods (Abstract)
        ----------
        set_display(self, show: bool)
            sets whether the display is to be shown or not


        MVC related abstract methods. Responses to change events broadcast by the model:

        graph_change(self, subject: SimulationModel)
            Respond to changes in graph (nodes/households network)

        model_state_change(self, subject: SimulationModel)
            Respond to changes in model state (e.g. running, extinct, timed-out)

        model_step_increment(self, subject: SimulationModel)
            Respond to single step increment in simulation

        model_simulation_stopped(self, subject: SimulationModel)
            Respond to simulation stopping
    """

    @abstractmethod
    def set_display(self, show: bool):
        """
        Sets whether this view be displayed or not.

            Parameters:
                show (bool): To display this view, set to True

            Returns:
                None
        """
        pass

    @abstractmethod
    def graph_change(self, subject: SimulationModel):
        """
        Respond to changes in graph (nodes/households network)

            Parameters:
                subject (SimulationModel): The simulation model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    @abstractmethod
    def model_state_change(self, subject: SimulationModel):
        """
        Respond to changes in model state (e.g. running, extinct, timed-out)

            Parameters:
                subject (SimulationModel): The simulation model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    @abstractmethod
    def model_step_increment(self, subject: SimulationModel):
        """
        Respond to single step increment in simulation

            Parameters:
                subject (SimulationModel): The simulation model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    @abstractmethod
    def model_simulation_stopped(self, subject: SimulationModel):
        """
        Respond to simulation stopping

            Parameters:
                subject (SimulationModel): The simulation model being displayed by this simulation view.

            Returns:
                None
        """
        pass
