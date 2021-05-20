from abc import ABC, abstractmethod

class SimulationView(ABC):

    @abstractmethod
    def set_display(self, show: bool):
        """ should the display be shown or not """
        pass

    """
        Simulation View interface (MVC pattern)
    """

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
