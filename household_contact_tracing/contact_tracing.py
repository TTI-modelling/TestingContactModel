from __future__ import annotations

from typing import List

from collections.abc import Callable

from household_contact_tracing.network import Network, TestType, Node, PositivePolicy


class ContactTracing:
    """ 'Context' class for contact tracing processes/strategies (Strategy pattern) """

    def __init__(self, network: Network, params: dict):
        self.network = network

        # Parameter Inputs:
        # contact tracing parameters
        self.household_positive_policy = PositivePolicy.lfa_testing_no_quarantine
        self.LFA_testing_requires_confirmatory_PCR = False
        self.node_daily_prob_lfa_test = 1

        # contact tracing functions (runtime updatable)
        self.prob_testing_positive_lfa_func = self.prob_testing_positive_lfa
        self.prob_testing_positive_pcr_func = self.prob_testing_positive_pcr

        # Update instance variables with anything in params
        for param_name in self.__dict__:
            if param_name in params:
                self.__dict__[param_name] = params[param_name]

    @property
    def prob_testing_positive_lfa_func(self) -> Callable[[int], float]:
        return self._prob_testing_positive_lfa_func

    @prob_testing_positive_lfa_func.setter
    def prob_testing_positive_lfa_func(self, fn: Callable[[int], float]):
        self._prob_testing_positive_lfa_func = fn

    @property
    def prob_testing_positive_pcr_func(self) -> Callable[[int], float]:
        return self._prob_testing_positive_pcr_func

    @prob_testing_positive_pcr_func.setter
    def prob_testing_positive_pcr_func(self, fn: Callable[[int], float]):
        self._prob_testing_positive_pcr_func = fn

    def prob_testing_positive_pcr(self, infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0
        else:
            return 0

    def prob_testing_positive_lfa(self, infectious_age):
        if infectious_age in [4, 5, 6]:
            return 1
        else:
            return 0

    def act_on_confirmatory_pcr_results(self, time: int):
        """Once on a individual receives a positive pcr result we need to act on it.

        This takes the form of:
        * Household members start lateral flow testing
        * Contact tracing is propagated
        """
        for node in self.network.all_nodes():
            if node.confirmatory_PCR_test_result_time == time:
                node.household.apply_positive_policy(time, self.household_positive_policy)

    def isolate_positive_lateral_flow_tests(self, time: int, positive_nodes: List[Node]):
        """A if a node tests positive on LFA, we assume that they isolate and stop LFA testing

        If confirmatory PCR testing is not required, then we do not start LFA testing the household at this point
        in time.
        """

        for node in positive_nodes:
            node.received_positive_test_result = True

            if node.will_uptake_isolation:
                node.isolated = True

            node.avenue_of_testing = TestType.lfa
            node.positive_test_time = time
            node.being_lateral_flow_tested = False

            if not node.household.applied_household_positive_policy and \
                    not self.LFA_testing_requires_confirmatory_PCR:
                node.household.apply_positive_policy(time, self.household_positive_policy)

    def confirmatory_pcr_test_LFA_nodes(self, time: int, positive_nodes: List[Node]):
        """Nodes who receive a positive LFA result will be tested using a PCR test."""
        for node in positive_nodes:
            if not node.taken_confirmatory_PCR_test:
                node.take_confirmatory_pcr_test(time, self.prob_testing_positive_pcr)

    def act_on_positive_LFA_tests(self, time: int):
        """For nodes who test positive on their LFA test, take the appropriate action depending
        on the policy
        """
        positive_nodes = self.lft_nodes(time)

        self.isolate_positive_lateral_flow_tests(time, positive_nodes)

        if self.LFA_testing_requires_confirmatory_PCR:
            self.confirmatory_pcr_test_LFA_nodes(time, positive_nodes)

    def lft_nodes(self, time: int) -> List[Node]:
        """Performs a days worth of lateral flow testing.

        Returns:
            A list of nodes who have tested positive through the lateral flow tests.
        """
        positive_nodes = []
        for node in self.network.all_nodes():
            if node.being_lateral_flow_tested:
                if node.will_lfa_test_today(self.node_daily_prob_lfa_test):
                    if not node.received_positive_test_result:
                        if node.lfa_test_node(time, self.prob_testing_positive_lfa_func):
                            positive_nodes.append(node)
        return positive_nodes
