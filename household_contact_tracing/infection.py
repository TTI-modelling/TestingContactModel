class InfectionBehaviourInterface:
    """ Interface for simulation infection behaviours """

    def increment_contact_tracing(self):
        pass


class Infection(InfectionBehaviourInterface):
    """ Branching Process Simulation Controller (MVC pattern) """

    def __init__(self, network):
        self._network = network

    def increment_infection(self):
        pass
