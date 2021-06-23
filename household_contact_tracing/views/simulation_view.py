from abc import ABC, abstractmethod


class SimulationView(ABC):
    """
        Simulation View (Abstract) (MVC pattern)
        This is the abstract parent of all view classes. Sub-class for adding new views.

        Methods (Abstract)
        ----------
        set_display(self, show: bool)
            sets whether the display is to be shown or not


        MVC related abstract methods. Responses to change events broadcast by the model:

        graph_change(self, subject)
            Respond to changes in graph (nodes/households network)

        model_state_change(self, subject)
            Respond to changes in model state (e.g. running, extinct, timed-out)

        model_step_increment(self, subject)
            Respond to single step increment in simulation

        model_simulation_stopped(self, subject)
            Respond to simulation stopping
    """

    @abstractmethod
    def set_display(self, show: bool):
        """ should the display be shown or not """
        pass

    @abstractmethod
    def graph_change(self, subject):
        """ Respond to changes in graph (nodes/households network) """
        pass

    @abstractmethod
    def model_state_change(self, subject):
        """ Respond to changes in model state (e.g. running, extinct, timed-out) """
        pass

    @abstractmethod
    def model_step_increment(self, subject):
        """ Respond to single step increment in simulation """
        pass

    @abstractmethod
    def model_simulation_stopped(self, subject):
        """ Respond to simulation stopping """
        pass
