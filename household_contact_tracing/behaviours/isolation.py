"""Methods implementing isolation for each model type.
These methods change the isolation state of Nodes and Households based on various node
and Household attributes."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable

from household_contact_tracing.network import Network, TestType, Household, EdgeType


class UpdateIsolation(ABC):
    def __init__(self, network: Network,
                 apply_policy_for_household_contacts_of_a_positive_case: Callable):
        self.network = network
        self.apply_policy_for_household_contacts_of_a_positive_case = apply_policy_for_household_contacts_of_a_positive_case

    @abstractmethod
    def update_isolation(self, time):
        """Isolate all non isolated households where the infection has been reported
         (excludes those who will not take up isolation if prob <1)"""
        pass

    def update_all_households_contact_traced(self, time):
        """Update the contact traced status for all households that have had the
         contact tracing process get there."""
        for household in self.network.all_households:
            if household.time_until_contact_traced <= time:
                if not household.contact_traced:
                    self.contact_trace_household(household, time)

    @abstractmethod
    def contact_trace_household(self, household: Household, time: int):
        """The things that are done when a Household is contact traced"""

    def update_network(self, household: Household):
        """
        When a house is contact traced, we need to place all the nodes under surveillance.
        If any of the nodes are symptomatic, we need to isolate the household.
        """
        # Update the house to the contact traced status
        household.contact_traced = True

        # Update the nodes to the contact traced status
        for node in household.nodes:
            node.contact_traced = True

        # Colour the edges within household
        self.network.label_edges_inside_household(household, EdgeType.within_house)

    def quarantine_traced_node(self, household):
        traced_node = self.find_traced_node(household)

        # the traced node should go into quarantine
        if not traced_node.isolated and traced_node.will_uptake_isolation:
            traced_node.isolated = True

    def find_traced_node(self, household: Household):
        """Work out which was the traced node."""
        tracing_household = household.being_contact_traced_from
        traced_node_id = self.network.get_edge_between_household(household, tracing_household)[0]
        return self.network.node(traced_node_id)


class UpdateIsolationHouseholdLevel(UpdateIsolation):
    def update_isolation(self, time: int):
        self.update_all_households_contact_traced(time)

        for node in self.network.all_nodes():
            if node.time_of_reporting + node.testing_delay == time:
                if not node.household.isolated:
                    if not node.household.contact_traced:
                        node.household.isolate_household(time)

    def contact_trace_household(self, household: Household, time: int):
        self.update_network(household)
        self.isolate_household_if_symptomatic_nodes(household, time)
        self.quarantine_traced_node(household)

    @staticmethod
    def isolate_household_if_symptomatic_nodes(household: Household, time: int):
        """If there are any symptomatic nodes in the household then isolate the household."""
        for node in household.nodes:
            if node.symptom_onset_time <= time and not node.completed_isolation:
                household.isolate_household(time)
                break


class UpdateIsolationIndividualLevelTracing(UpdateIsolation):
    def update_isolation(self, time: int):
        self.update_all_households_contact_traced(time)

        for node in self.network.all_nodes():
            if node.time_of_reporting + node.testing_delay == time:
                if node.received_positive_test_result:
                    if not node.household.isolated:
                        if not node.household.contact_traced:
                            node.household.isolate_household(time)

    def contact_trace_household(self, household: Household, time: int):
        self.update_network(household)
        self.quarantine_traced_node(household)


class UpdateIsolationIndividualTracingDailyTesting(UpdateIsolation):
    def update_isolation(self, time: int):
        self.update_all_households_contact_traced(time)

        for node in self.network.all_nodes():
            if node.positive_test_time == time:
                if node.avenue_of_testing == TestType.pcr:
                    if node.received_positive_test_result:
                        if not node.household.applied_policy_for_household_contacts_of_a_positive_case:
                            self.apply_policy_for_household_contacts_of_a_positive_case(node.household, time)

    def contact_trace_household(self, household: Household, time: int):
        self.update_network(household)
        traced_node = self.find_traced_node(household)
        # the traced node is now being lateral flow tested
        if traced_node.node_will_take_up_lfa_testing:
            if not traced_node.received_positive_test_result:
                traced_node.being_lateral_flow_tested = True
                traced_node.time_started_lfa_testing = time
