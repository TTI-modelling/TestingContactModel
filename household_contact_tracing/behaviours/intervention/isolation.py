from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Callable

from household_contact_tracing.network import Network, TestType, Node
from household_contact_tracing.parameterised import Parameterised


class Isolation(ABC, Parameterised):
    """
        An abstract base class used to represent the highest level Isolation behaviour.

        Note:   This class forms part of a 'Strategy' pattern. All child classes implement a family of possible
                behaviours or strategies (ways of isolating).
                Add further child classes to add new behaviour types (strategies) that can be selected and updated at
                design or run-time.

        Attributes
        ----------
        network: ContactTracingNetwork
            The store of Nodes and households used in the simulation
        household_positive_policy = PositivePolicy.lfa_testing_no_quarantine
            The policy on positive nodes
        LFA_testing_requires_confirmatory_PCR = False
            Whether or not the testing requires a confirmatory PCR

        Methods
        -------

        update_isolation(self, time: int)
            Increments the intervention process by one step, performing any steps required due to the current step
            number (time)

        isolate_self_reporting_cases(self, time: int)
            Applies the intervention status to nodes who have reached their self-report time.

        update_households_contact_traced(self, time: int)
            Update the contact traced status for all households that have had the contact tracing process get there.

    """

    def __init__(self, network: Network, params: dict):
        self.network = network
        self.household_positive_policy = "lfa_testing_no_quarantine"
        self.LFA_testing_requires_confirmatory_PCR = False

        self.update_params(params)

    @abstractmethod
    def isolate_self_reporting_cases(self, time: int):
        """Applies the intervention status to nodes who have reached their self-report time.

        Arguments:
            time -- The current step number (e.g. day) of the simulation

        Returns:
            None
        """

    @abstractmethod
    def update_households_contact_traced(self, time: int):
        """Update the contact traced status for all households that have had the contact tracing process get there.

         Arguments:
            time -- The current step number (e.g. day) of the simulation

        Returns:
            None
        """

    @abstractmethod
    def update_isolation(self, time: int):
        """ Increments the intervention process by one step, performing any steps required due to the current step
            number (time)

        Arguments:
            time -- The current step number (e.g. day) of the simulation

        Returns:
            None
        """


class HouseholdIsolation(Isolation):

    def isolate_self_reporting_cases(self, time: int):
        """Applies the intervention status to nodes who have reached their self-report time.
        They may of course decide to not adhere to said intervention, or may be a member of a household
        who will not uptake intervention
        """
        for node in self.network.all_nodes():
            if node.tracing_adherence.will_uptake_isolation:
                if node.time_of_reporting == time:
                    node.isolated = True

    def update_households_contact_traced(self, time: int):
        """Update the contact traced status for all households that have had the
         contact tracing process get there."""
        for household in self.network.all_households:
            if household.time_until_contact_traced <= time:
                if not household.contact_traced:
                    household.update_network()
                    household.isolate_if_symptomatic_nodes(time)
                    household.quarantine_traced_node()

    def update_isolation(self, time: int):
        for node in self.network.all_nodes():
            if node.time_of_reporting + node.testing_delay == time:
                if not node.household.isolated:
                    if not node.household.contact_traced:
                        node.household.isolate_household(time)


class IndividualIsolation(HouseholdIsolation):
    def update_households_contact_traced(self, time: int):
        """Update the contact traced status for all households that have had the
         contact tracing process get there."""
        for household in self.network.all_households:
            if household.time_until_contact_traced <= time:
                if not household.contact_traced:
                    household.update_network()
                    household.quarantine_traced_node()

    def update_isolation(self, time: int):
        for node in self.network.all_nodes():
            if node.time_of_reporting + node.testing_delay == time:
                if node.received_positive_test_result:
                    if not node.household.isolated:
                        if not node.household.contact_traced:
                            node.household.isolate_household(time)


class DailyTestingIsolation(HouseholdIsolation):

    def __init__(self, network: Network, params: dict):
        super().__init__(network, params)
        self._prob_pcr_positive = None

    @property
    def prob_pcr_positive(self) -> Callable[[int], float]:
        return self._prob_pcr_positive

    @prob_pcr_positive.setter
    def prob_pcr_positive(self, fn: Callable[[int], float]):
        self._prob_pcr_positive = fn

    def update_households_contact_traced(self, time: int):
        """Update the contact traced status for all households that have had the
         contact tracing process get there."""
        for household in self.network.all_households:
            if household.time_until_contact_traced <= time:
                if not household.contact_traced:
                    household.update_network()
                    traced_node = household.find_traced_node()
                    # the traced node is now being lateral flow tested
                    if traced_node.lfd_testing_adherence.node_will_take_up_lfa_testing:
                        if not traced_node.received_positive_test_result:
                            traced_node.being_lateral_flow_tested = True
                            traced_node.lfd_testing.time_started_lfa_testing = time

    def update_isolation(self, time: int):
        for node in self.network.all_nodes():
            if node.lfd_testing.positive_test_time == time:
                if node.lfd_testing.avenue_of_testing == TestType.pcr:
                    if node.received_positive_test_result:
                        if not node.household.applied_household_positive_policy:
                            node.household.apply_positive_policy(time, self.household_positive_policy)

    def act_on_confirmatory_pcr_results(self, time: int):
        """Once on a individual receives a positive pcr result we need to act on it.

        This takes the form of:
        * Household members start lateral flow testing
        * Contact tracing is propagated
        """
        for node in self.network.all_nodes():
            if node.lfd_testing.confirmatory_PCR_test_result_time == time:
                node.household.apply_positive_policy(time, self.household_positive_policy)

    def isolate_positive_lateral_flow_tests(self, time: int, positive_nodes: List[Node]):
        """A if a node tests positive on LFA, we assume that they isolate and stop LFA testing

        If confirmatory PCR testing is not required, then we do not start LFA testing the household at this point
        in time.
        """

        for node in positive_nodes:
            node.received_positive_test_result = True

            if node.tracing_adherence.will_uptake_isolation:
                node.isolated = True

            node.avenue_of_testing = TestType.lfa
            node.lfd_testing.positive_test_time = time
            node.being_lateral_flow_tested = False

            if not node.household.applied_household_positive_policy and \
                    not self.LFA_testing_requires_confirmatory_PCR:
                node.household.apply_positive_policy(time, self.household_positive_policy)

    def act_on_positive_LFA_tests(self, time: int, positive_nodes: List[Node]):
        """For nodes who test positive on their LFA test, take the appropriate action depending
        on the policy
        """
        self.isolate_positive_lateral_flow_tests(time, positive_nodes)

        if self.LFA_testing_requires_confirmatory_PCR:
            self.confirmatory_pcr_test_LFA_nodes(time, positive_nodes)

    def confirmatory_pcr_test_LFA_nodes(self, time: int, positive_nodes: List[Node]):
        """Nodes who receive a positive LFA result will be tested using a PCR test."""
        for node in positive_nodes:
            if not node.lfd_testing.taken_confirmatory_PCR_test:
                node.take_confirmatory_pcr_test(time, self.prob_pcr_positive)
