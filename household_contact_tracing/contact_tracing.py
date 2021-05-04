import numpy as np
import numpy.random as npr

from household_contact_tracing.network import Household, EdgeType


class ContactTracing:
    """ Class for contract tracing  """

    def __init__(self, network, params):
        self._network = network

        self._contact_trace_household = None
        self._increment = None

        # Parameter Inputs:
        # contact tracing parameters
        self.contact_tracing_success_prob = params["contact_tracing_success_prob"]
        if "do_2_step" in params:
            self.do_2_step = params["do_2_step"]
        else:
            self.do_2_step = False
        if "prob_has_trace_app" in params:
            self.prob_has_trace_app = params["prob_has_trace_app"]
        else:
            self.prob_has_trace_app = 0
        if "hh_propensity_to_use_trace_app" in params:
            self.hh_propensity_to_use_trace_app = params["hh_propensity_to_use_trace_app"]
        else:
            self.hh_propensity_to_use_trace_app = 1
        if "test_before_propagate_tracing" in params:
            self.test_before_propagate_tracing = params["test_before_propagate_tracing"]
        else:
            self.test_before_propagate_tracing = True
        self.test_delay = params["test_delay"]
        self.contact_trace_delay = params["contact_trace_delay"]

    @property
    def network(self):
        return self._network

    @property
    def contact_trace_household(self):
        return self._contact_trace_household

    @contact_trace_household.setter
    def contact_trace_household(self, fn):
        self._contact_trace_household = fn

    @property
    def increment(self):
        return self._increment

    @increment.setter
    def increment(self, fn):
        self._increment = fn

    def hh_propensity_use_trace_app(self) -> bool:
        if npr.binomial(1, self.hh_propensity_to_use_trace_app) == 1:
            return True
        else:
            return False

    def has_contact_tracing_app(self) -> bool:
        return npr.binomial(1, self.prob_has_trace_app) == 1

    def testing_delay(self) -> int:
        if self.test_before_propagate_tracing is False:
            return 0
        else:
            return round(self.test_delay)
