from abc import ABC, abstractmethod

from household_contact_tracing.network import Network, Household, EdgeType


class ContactTraceHouseholdBehaviour(ABC):

    def __init__(self, network: Network):
        self._network = network

    @abstractmethod
    def contact_trace_household(self, household: Household, time: int):
        pass

    def update_network(self, household: Household):
        """
        When a house is contact traced, we need to place all the nodes under surveillance.
        If any of the nodes are symptomatic, we need to isolate the household.
        """
        # Update the house to the contact traced status
        household.contact_traced = True

        # Update the nodes to the contact traced status
        for node in household.nodes():
            node.contact_traced = True

        # Colour the edges within household
        for edge in household.within_house_edges:
            self._network.graph.edges[edge[0], edge[1]].update({"edge_type": EdgeType.within_house.name})

    def quarantine_traced_node(self, household):
        traced_node = self.find_traced_node(household)

        # the traced node should go into quarantine
        if not traced_node.isolated and traced_node.will_uptake_isolation:
            traced_node.isolated = True

    def find_traced_node(self, household):
        # work out which was the traced node
        tracing_household = self._network.houses.household(household.being_contact_traced_from)
        traced_node_id = self._network.get_edge_between_household(household, tracing_household)[0]
        return self._network.node(traced_node_id)

    def isolate_household(self, household: Household, time: int):
        """Isolates a house so that all infectives in that household may no longer infect others.

        If the house is being surveillance due to a successful contact trace, and not due to
        reporting symptoms, update the edge label to display this.

        For households that were connected to this household, they are assigned a time until
        contact traced

        When a house has been contact traced, all nodes in the house are under surveillance for
        symptoms. When a node becomes symptomatic, the house moves to isolation status.
        """

        # Makes sure the isolate household is never applied multiple times to the same household
        if household.isolated:
            return

        # update isolated and contact traced status for Household
        household.contact_traced = True
        household.isolated = True
        household.isolated_time = time

        # Update isolated and contact traced status for Nodes in Household
        for node in household.nodes():
            node.contact_traced = True
            if node.will_uptake_isolation:
                node.isolated = True

        # Work out which house started the contact trace that led to this house being isolated,
        # in order to label this on the graph.
        # A household may be being isolated because someone in the household self reported symptoms,
        # so `being_contact_traced_from` may be None.
        if household.being_contact_traced_from is not None:
            house_which_contact_traced = self._network.houses.household(household.being_contact_traced_from)

            # Initially the edge is assigned the contact tracing label, may be updated if the
            # contact tracing does not succeed
            if self._network.is_edge_app_traced(self._network.get_edge_between_household(household, house_which_contact_traced)):
                self._network.label_edges_between_houses(household, house_which_contact_traced,
                                                         EdgeType.app_traced.name)
            else:
                self._network.label_edges_between_houses(household, house_which_contact_traced,
                                                         EdgeType.between_house.name)

        # We update the label of every edge so that we can tell which household have been contact
        # traced when we visualise
        for edge in household.within_house_edges:
            self._network.graph.edges[edge[0], edge[1]].update({"edge_type": EdgeType.within_house.name})


class ContactTraceHouseholdLevel(ContactTraceHouseholdBehaviour):
    def contact_trace_household(self, household: Household, time: int):
        self.update_network(household)
        self.isolate_household_if_symptomatic_nodes(household, time)
        self.quarantine_traced_node(household)

    def isolate_household_if_symptomatic_nodes(self, household: Household, time: int):
        symptomatic_nodes = [node for node in household.nodes() if
                             node.symptom_onset_time <= time and not node.completed_isolation]
        if symptomatic_nodes:
            self.isolate_household(household, time)


class ContactTraceHouseholdIndividualLevel(ContactTraceHouseholdBehaviour):
    def contact_trace_household(self, household: Household, time: int):
        self.update_network(household)
        self.quarantine_traced_node(household)


class ContactTraceHouseholdIndividualTracingDailyTest(ContactTraceHouseholdBehaviour):
    def contact_trace_household(self, household: Household, time: int):
        self.update_network(household)
        traced_node = self.find_traced_node(household)
        # the traced node is now being lateral flow tested
        if traced_node.node_will_take_up_lfa_testing and not traced_node.received_positive_test_result:
            traced_node.being_lateral_flow_tested = True
            traced_node.time_started_lfa_testing = time
