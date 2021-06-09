from __future__ import annotations

from typing import List, Callable

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

        # Update instance variables with anything in params
        for param_name in self.__dict__:
            if param_name in params:
                self.__dict__[param_name] = params[param_name]

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

    def confirmatory_pcr_test_LFA_nodes(self, time: int, positive_nodes: List[Node],
                                        prob_pcr_positive: Callable):
        """Nodes who receive a positive LFA result will be tested using a PCR test."""
        for node in positive_nodes:
            if not node.taken_confirmatory_PCR_test:
                node.take_confirmatory_pcr_test(time, prob_pcr_positive)

    def act_on_positive_LFA_tests(self, time: int, prob_pcr_positive: Callable,
                                  prob_lfa_positive: Callable):
        """For nodes who test positive on their LFA test, take the appropriate action depending
        on the policy
        """
        positive_nodes = self.lft_nodes(time, prob_lfa_positive)

        self.isolate_positive_lateral_flow_tests(time, positive_nodes)

        if self.LFA_testing_requires_confirmatory_PCR:
            self.confirmatory_pcr_test_LFA_nodes(time, positive_nodes, prob_pcr_positive)

    def lft_nodes(self, time: int, prob_lfa_positive: Callable) -> List[Node]:
        """Performs a days worth of lateral flow testing.

        Returns:
            A list of nodes who have tested positive through the lateral flow tests.
        """
        positive_nodes = []
        for node in self.network.all_nodes():
            if node.being_lateral_flow_tested:
                if node.will_lfa_test_today(self.node_daily_prob_lfa_test):
                    if not node.received_positive_test_result:
                        if node.lfa_test_node(time, prob_lfa_positive):
                            positive_nodes.append(node)
        return positive_nodes
