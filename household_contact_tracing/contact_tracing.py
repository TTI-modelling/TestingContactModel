class ContactTracing:
    """ Class for contract tracing  """

    def __init__(self, network, params):
        self._network = network

    @property
    def network(self):
        return self._network

    def increment(self):
        pass
