from __future__ import annotations

import numpy as np
from collections.abc import Callable

from household_contact_tracing.network import Network, Household, TestType, Node


class ContactTracing:
    """ 'Context' class for contact tracing processes/strategies (Strategy pattern) """

    def __init__(self, network: Network, params: dict):
        self.network = network

        # Parameter Inputs:
        # contact tracing parameters
        self.policy_for_household_contacts_of_a_positive_case = 'lfa testing no quarantine'
        self.LFA_testing_requires_confirmatory_PCR = False
        self.node_daily_prob_lfa_test = 1

        # contact tracing functions (runtime updatable)
        self.prob_testing_positive_lfa_func = self.prob_testing_positive_lfa
        self.prob_testing_positive_pcr_func = self.prob_testing_positive_pcr

        self.current_LFA_positive_nodes = []

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

    def apply_policy_for_household_contacts_of_a_positive_case(self, household: Household,
                                                               time: int):
        """We apply different policies to the household contacts of a discovered case.
        The policy is set using the policy_for_household_contacts_of_a_positive_case input.

        Available policy settings:
            * "lfa testing no quarantine" - Household contacts start LFA testing, but do not quarantine unless they develop symptoms
            * "lfa testing and quarantine" - Household contacts start LFA testing, and quarantine.
            * "no lfa testing and quarantine" - Household contacts do not start LFA testing, quarantine. They will book a PCR test if they develop symptoms.
        """

        # set the household attributes to declare that we have already applied the policy
        household.applied_policy_for_household_contacts_of_a_positive_case = True

        if self.policy_for_household_contacts_of_a_positive_case == 'lfa testing no quarantine':
            self.start_lateral_flow_testing_household(household, time)
        elif self.policy_for_household_contacts_of_a_positive_case == 'lfa testing and quarantine':
            self.start_lateral_flow_testing_household_and_quarantine(household, time)
        elif self.policy_for_household_contacts_of_a_positive_case == 'no lfa testing only quarantine':
            household.isolate_household(time)
        else:
            raise Exception("""policy_for_household_contacts_of_a_positive_case not recognised. Must be one of the 
            following options:
                * "lfa testing no quarantine"
                * "lfa testing and quarantine"
                * "no lfa testing only quarantine" """)

    @staticmethod
    def start_lateral_flow_testing_household(household: Household, time: int):
        """Sets the household to the lateral flow testing status so that new within household
        infections are tested

        All nodes in the household start lateral flow testing

        Args:
            household (Household): The household which is initiating testing
        """

        household.being_lateral_flow_tested = True
        household.being_lateral_flow_tested_start_time = time

        for node in household.nodes:
            if node.node_will_take_up_lfa_testing and not node.received_positive_test_result and \
                    not node.being_lateral_flow_tested:
                node.being_lateral_flow_tested = True
                node.time_started_lfa_testing = time

    @staticmethod
    def start_lateral_flow_testing_household_and_quarantine(household: Household, time: int):
        """Sets the household to the lateral flow testing status so that new within household
        infections are tested

        All nodes in the household start lateral flow testing and start quarantining

        Args:
            household (Household): The household which is initiating testing
        """
        household.being_lateral_flow_tested = True
        household.being_lateral_flow_tested_start_time = time
        household.isolated = True
        household.isolated_time = True
        household.contact_traced = True

        for node in household.nodes:
            if node.node_will_take_up_lfa_testing and not node.received_positive_test_result and \
                    not node.being_lateral_flow_tested:
                node.being_lateral_flow_tested = True
                node.time_started_lfa_testing = time

            if node.will_uptake_isolation:
                node.isolated = True

    def lfa_test_node(self, node: Node, time: int):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (ContactTracingNode): The node to be tested today
        """

        infectious_age = time - node.time_infected

        prob_positive_result = self.prob_testing_positive_lfa_func(infectious_age)

        if np.random.binomial(1, prob_positive_result) == 1:
            return True
        else:
            return False

    def will_lfa_test_today(self, node: Node) -> bool:

        if node.propensity_to_miss_lfa_tests:

            if np.random.binomial(1, self.node_daily_prob_lfa_test) == 1:
                return True
            else:
                return False
        else:
            return True

    def act_on_confirmatory_pcr_results(self, time: int):
        """Once on a individual receives a positive pcr result we need to act on it.

        This takes the form of:
        * Household members start lateral flow testing
        * Contact tracing is propagated
        """
        for node in self.network.all_nodes():
            if node.confirmatory_PCR_test_result_time == time:
                self.apply_policy_for_household_contacts_of_a_positive_case(node.household, time)

    def get_positive_lateral_flow_nodes(self, time: int):
        """Performs a days worth of lateral flow testing.

        Returns:
            List[ContactTracingNode]: A list of nodes who have tested positive through the lateral flow tests.
        """

        return [
            node for node in self.network.all_nodes()
            if node.being_lateral_flow_tested
               and self.will_lfa_test_today(node)
               and not node.received_positive_test_result
               and self.lfa_test_node(node, time)
        ]

    def isolate_positive_lateral_flow_tests(self, time: int):
        """A if a node tests positive on LFA, we assume that they isolate and stop LFA testing

        If confirmatory PCR testing is not required, then we do not start LFA testing the household at this point
        in time.
        """

        for node in self.current_LFA_positive_nodes:
            node.received_positive_test_result = True

            if node.will_uptake_isolation:
                node.isolated = True

            node.avenue_of_testing = TestType.lfa
            node.positive_test_time = time
            node.being_lateral_flow_tested = False

            if not node.household.applied_policy_for_household_contacts_of_a_positive_case and \
                    not self.LFA_testing_requires_confirmatory_PCR:
                self.apply_policy_for_household_contacts_of_a_positive_case(node.household, time)

    def take_confirmatory_pcr_test(self, node: Node, time: int):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (ContactTracingNode): The node to be tested today
        """

        infectious_age_when_tested = time - node.time_infected
        prob_positive_result = self.prob_testing_positive_pcr_func(infectious_age_when_tested)

        node.confirmatory_PCR_test_time = time
        node.confirmatory_PCR_test_result_time = time + node.testing_delay
        node.taken_confirmatory_PCR_test = True

        if np.random.binomial(1, prob_positive_result) == 1:
            node.confirmatory_PCR_result_was_positive = True

        else:
            node.confirmatory_PCR_result_was_positive = False

    def confirmatory_pcr_test_LFA_nodes(self, time: int):
        """Nodes who receive a positive LFA result will be tested using a PCR test."""
        for node in self.current_LFA_positive_nodes:
            if not node.taken_confirmatory_PCR_test:
                self.take_confirmatory_pcr_test(node, time)

    def act_on_positive_LFA_tests(self, time: int):
        """For nodes who test positive on their LFA test, take the appropriate action depending
        on the policy
        """
        self.current_LFA_positive_nodes = self.get_positive_lateral_flow_nodes(time)

        self.isolate_positive_lateral_flow_tests(time)

        if self.LFA_testing_requires_confirmatory_PCR:
            self.confirmatory_pcr_test_LFA_nodes(time)
