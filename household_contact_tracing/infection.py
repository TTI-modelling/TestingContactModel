class InfectionBehaviourInterface:
    """ Interface for simulation infection processes """

    def increment_contact_tracing(self):
        pass


class Infection(InfectionBehaviourInterface):
    """ Class for Infection processes """

    def __init__(self, network):
        self._network = network

    def increment_infection(self):
        pass
