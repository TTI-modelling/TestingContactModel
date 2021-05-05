import numpy as np
import numpy.random as npr

from household_contact_tracing.network import Network, Household, EdgeType


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
    def contact_trace_household(self) -> 'ContactTraceHouseholdBehaviour':
        return self._contact_trace_household

    @contact_trace_household.setter
    def contact_trace_household(self, contact_trace_household: 'ContactTraceHouseholdBehaviour'):
        self._contact_trace_household = contact_trace_household

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


class ContactTraceHouseholdBehaviour:

    def contact_trace_household(self, household: Household, network: Network, time: int):
        pass

    def update_network(self, household: Household, network: Network):
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
        [
            network.graph.edges[edge[0], edge[1]].update({"edge_type": EdgeType.within_house.name})
            for edge in household.within_house_edges
        ]

    def quarantine_traced_node(self, household, network: Network):
        traced_node = self.find_traced_node(household, network)

        # the traced node should go into quarantine
        if not traced_node.isolated and traced_node.will_uptake_isolation:
            #Todo: AG: Peter/Martyn checking 2nd operand is traced_node and not node (as before)
            traced_node.isolated = True

    def find_traced_node(self, household, network: Network):
        # work out which was the traced node
        tracing_household = network.houses.household(household.being_contact_traced_from)
        traced_node_id = network.get_edge_between_household(household, tracing_household)[0]
        return network.node(traced_node_id)

    def isolate_household_if_symptomatic_nodes(self, household: Household, network:Network, time: int):
        symptomatic_nodes = [node for node in household.nodes() if
                             node.symptom_onset_time <= time and not node.completed_isolation]
        if symptomatic_nodes:
            self.isolate_household(household, network, time)

    def isolate_household(self, household: Household, network: Network, time: int):
        """
        Isolates a house so that all infectives in that household may no longer infect others.

        If the house is being surveillance due to a successful contact trace, and not due to reporting symptoms,
        update the edge label to display this.

        For households that were connected to this household, they are assigned a time until contact traced

        When a house has been contact traced, all nodes in the house are under surveillance for symptoms. When a node becomes symptomatic, the house moves to isolation status.
        """

        # Makes sure the isolate household is never applied multiple times to the same household
        if not household.isolated:

            # update the household and all nodes in the household to the contact traced status
            household.contact_traced = True
            for node in household.nodes():
                node.contact_traced = True

            # Households have a probability to take up isolation if traced

            # The house moves to isolated status if it has been assigned to take up isolation if trace, given a probability
            household.isolated = True
            # household.contact_traced = True
            household.isolated_time = time

            # Update every node in the house to the isolated status
            for node in household.nodes():
                if node.will_uptake_isolation:
                    node.isolated = True

            # Which house started the contact trace that led to this house being isolated, if there is one
            # A household may be being isolated because someone in the household self reported symptoms
            # Hence sometimes there is a None value for House which contact traced
            if household.being_contact_traced_from is not None:
                house_which_contact_traced = network.houses.household(household.being_contact_traced_from)

                # Initially the edge is assigned the contact tracing label, may be updated if the contact tracing does not succeed
                if network.is_edge_app_traced(
                        network.get_edge_between_household(household, house_which_contact_traced)):
                    self.label_node_edges_between_houses(network, household, house_which_contact_traced,
                                                         EdgeType.app_traced.name)
                else:
                    self.label_node_edges_between_houses(network, household, house_which_contact_traced,
                                                         EdgeType.between_house.name)

                    # We update the label of every edge so that we can tell which household have been contact traced when we visualise
            [
                network.graph.edges[edge[0], edge[1]].update({"edge_type": EdgeType.within_house.name})
                for edge in household.within_house_edges
            ]

    def label_node_edges_between_houses(self, network: Network, house_to: Household, house_from: Household,
                                        new_edge_type):
        # Annoying bit of logic to find the edge and label it
        for node_1 in house_to.nodes():
            for node_2 in house_from.nodes():
                if network.graph.has_edge(node_1.node_id, node_2.node_id):
                    network.graph.edges[node_1.node_id, node_2.node_id].update({"edge_type": new_edge_type})


class ContactTraceHouseholdBP(ContactTraceHouseholdBehaviour):
    def contact_trace_household(self, household: Household, network: Network, time: int):
        self.update_network(household, network)
        self.isolate_household_if_symptomatic_nodes(household, network, time)
        self.quarantine_traced_node(household, network)


class ContactTraceHouseholdUK(ContactTraceHouseholdBehaviour):
    def contact_trace_household(self, household: Household, network: Network, time: int):
        self.update_network(household, network)
        self.quarantine_traced_node(household)


class ContactTraceHouseholdContactModelTest(ContactTraceHouseholdBehaviour):
    def contact_trace_household(self, household: Household, network: Network, time: int):
        self.update_network(household, network)
        traced_node = self.find_traced_node(household, network)
        # the traced node is now being lateral flow tested
        if traced_node.node_will_take_up_lfa_testing and not traced_node.received_positive_test_result:
            traced_node.being_lateral_flow_tested = True
            traced_node.time_started_lfa_testing = time
