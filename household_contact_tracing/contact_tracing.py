class ContactTracingBehaviourInterface:
    """ Interface for simulation contact tracing """

    def increment_contact_tracing(self):
        pass


class ContactTracing(ContactTracingBehaviourInterface):
    """ Class for contract tracing  """

    def __init__(self, network):
        self._network = network

    def increment_contact_tracing(self):
        pass
