"""These methods provide various ways of implementing contact tracing."""

from abc import ABC, abstractmethod

from household_contact_tracing.network import Network, Household, EdgeType


class ContactTraceHousehold(ABC):

    def __init__(self, network: Network):
        self._network = network

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
        self._network.label_edges_inside_household(household, EdgeType.within_house)

    def quarantine_traced_node(self, household):
        traced_node = self.find_traced_node(household)

        # the traced node should go into quarantine
        if not traced_node.isolated and traced_node.will_uptake_isolation:
            traced_node.isolated = True

    def find_traced_node(self, household: Household):
        """Work out which was the traced node."""
        tracing_household = household.being_contact_traced_from
        traced_node_id = self._network.get_edge_between_household(household, tracing_household)[0]
        return self._network.node(traced_node_id)


class ContactTraceHouseholdLevel(ContactTraceHousehold):
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


class ContactTraceHouseholdIndividualLevel(ContactTraceHousehold):
    def contact_trace_household(self, household: Household, time: int):
        self.update_network(household)
        self.quarantine_traced_node(household)


class ContactTraceHouseholdIndividualTracingDailyTest(ContactTraceHousehold):
    def contact_trace_household(self, household: Household, time: int):
        self.update_network(household)
        traced_node = self.find_traced_node(household)
        # the traced node is now being lateral flow tested
        if traced_node.node_will_take_up_lfa_testing:
            if not traced_node.received_positive_test_result:
                traced_node.being_lateral_flow_tested = True
                traced_node.time_started_lfa_testing = time
