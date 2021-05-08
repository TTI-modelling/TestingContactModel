import numpy as np
import numpy.random as npr

from household_contact_tracing.network import Network, Household, EdgeType


class ContactTracing:
    """ Class for contract tracing  """

    def __init__(self, network: Network, params: dict):
        self._network = network

        self._contact_trace_household = None
        self._increment = None

        # Parameter Inputs:
        # contact tracing parameters
        self.contact_tracing_success_prob = params["contact_tracing_success_prob"]

        self.do_2_step = False
        self.prob_has_trace_app = 0
        self.hh_propensity_to_use_trace_app = 1
        self.test_before_propagate_tracing = True
        self.test_delay = 1
        self.contact_trace_delay = 1

        # Update instance variables with anything in params
        for param_name in self.__dict__:
            if param_name in params:
                self.__dict__[param_name] = params[param_name]

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

    def __init__(self, network: Network):
        self._network = network

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
        [
            self._network.graph.edges[edge[0], edge[1]].update({"edge_type": EdgeType.within_house.name})
            for edge in household.within_house_edges
        ]

    def quarantine_traced_node(self, household):
        traced_node = self.find_traced_node(household)

        # the traced node should go into quarantine
        if not traced_node.isolated and traced_node.will_uptake_isolation:
            #Todo: AG: Peter/Martyn checking 2nd operand is traced_node and not node (as before)
            traced_node.isolated = True

    def find_traced_node(self, household):
        # work out which was the traced node
        tracing_household = self._network.houses.household(household.being_contact_traced_from)
        traced_node_id = self._network.get_edge_between_household(household, tracing_household)[0]
        return self._network.node(traced_node_id)

    def isolate_household_if_symptomatic_nodes(self, household: Household, time: int):
        symptomatic_nodes = [node for node in household.nodes() if
                             node.symptom_onset_time <= time and not node.completed_isolation]
        if symptomatic_nodes:
            self.isolate_household(household, time)

    def isolate_household(self, household: Household, time: int):
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
                house_which_contact_traced = self._network.houses.household(household.being_contact_traced_from)

                # Initially the edge is assigned the contact tracing label, may be updated if the contact tracing does not succeed
                if self._network.is_edge_app_traced(
                        self._network.get_edge_between_household(household, house_which_contact_traced)):
                    self.label_node_edges_between_houses(household, house_which_contact_traced,
                                                         EdgeType.app_traced.name)
                else:
                    self.label_node_edges_between_houses(household, house_which_contact_traced,
                                                         EdgeType.between_house.name)

                    # We update the label of every edge so that we can tell which household have been contact traced when we visualise
            [
                self._network.graph.edges[edge[0], edge[1]].update({"edge_type": EdgeType.within_house.name})
                for edge in household.within_house_edges
            ]

    # Todo: For Peter -> Network
    def label_node_edges_between_houses(self, house_to: Household, house_from: Household, new_edge_type):
        # Annoying bit of logic to find the edge and label it
        for node_1 in house_to.nodes():
            for node_2 in house_from.nodes():
                if self._network.graph.has_edge(node_1.node_id, node_2.node_id):
                    self._network.graph.edges[node_1.node_id, node_2.node_id].update({"edge_type": new_edge_type})


class ContactTraceHousehold(ContactTraceHouseholdBehaviour):
    def contact_trace_household(self, household: Household, time: int):
        self.update_network(household)
        self.isolate_household_if_symptomatic_nodes(household, time)
        self.quarantine_traced_node(household)


class ContactTraceHouseholdUK(ContactTraceHouseholdBehaviour):
    def contact_trace_household(self, household: Household, time: int):
        self.update_network(household)
        self.quarantine_traced_node(household)


class ContactTraceHouseholdContactModelTest(ContactTraceHouseholdBehaviour):
    def contact_trace_household(self, household: Household, time: int):
        self.update_network(household)
        traced_node = self.find_traced_node(household)
        # the traced node is now being lateral flow tested
        if traced_node.node_will_take_up_lfa_testing and not traced_node.received_positive_test_result:
            traced_node.being_lateral_flow_tested = True
            traced_node.time_started_lfa_testing = time


class IncrementContactTracingBehaviour:
    def __init__(self, network: Network, contact_tracing: ContactTracing):
        self._network = network
        self._contact_tracing = contact_tracing

    def increment_contact_tracing(self, time: int):
        pass


class IncrementContactTracingHousehold(IncrementContactTracingBehaviour):

    def increment_contact_tracing(self, time: int):
        """
        Performs a days worth of contact tracing by:
        * Looks for houses where the contact tracing delay is over and moves them to the contact traced state
        * Looks for houses in the contact traced state, and checks them for symptoms. If any of them have symptoms, the house is isolated

        The isolation function also assigns contact tracing times to any houses that had contact with that household

        For each node that is contact traced
        """

        # Isolate all households under observation that now display symptoms (excludes those who will not take up isolation if prob <1)
        [
            self._contact_tracing.contact_trace_household.isolate_household(node.household(), time)
            for node in self._network.all_nodes()
            if node.symptom_onset_time <= time
            and node.contact_traced
            and not node.isolated
            and not node.completed_isolation
        ]

        # Look for houses that need to propagate the contact tracing because their test result has come back
        # Necessary conditions: household isolated, symptom onset + testing delay = time

        # Propagate the contact tracing for all households that self-reported and have had their test results come back
        [
            self.propagate_contact_tracing(node.household(), time)
            for node in self._network.all_nodes()
            if node.time_of_reporting + node.testing_delay == time
            and not node.household().propagated_contact_tracing
        ]

        # Propagate the contact tracing for all households that are isolated due to exposure, have developed symptoms and had a test come back
        [
            self.propagate_contact_tracing(node.household(), time)
            for node in self._network.all_nodes()
            if node.symptom_onset_time <= time
            and not node.household().propagated_contact_tracing
            and node.household().isolated_time + node.testing_delay <= time
        ]

        # Update the contact tracing index of households
        # That is, re-evaluate how far away they are from a known infected case (traced + symptom_onset_time + testing_delay)
        self.update_contact_tracing_index()

        if self._contact_tracing.do_2_step:
            # Propagate the contact tracing from any households with a contact tracing index of 1
            [
                self.propagate_contact_tracing(household, time)
                for household in self._network.houses.all_households()
                if household.contact_tracing_index == 1
                and not household.propagated_contact_tracing
                and household.isolated
            ]

    def propagate_contact_tracing(self, household: Household, time: int):
        """
        To be called after a node in a household either reports their symptoms, and gets tested, when a household that is under surveillance develops symptoms + gets tested.
        """
        # update the propagation data
        household.propagated_contact_tracing = True
        household.time_propagated_tracing = time

        # Contact tracing attempted for the household that infected the household currently propagating the infection

        infected_by = household.infected_by()

        # If infected by = None, then it is the origin node, a special case
        if infected_by and not infected_by.isolated:
            self.attempt_contact_trace_of_household(infected_by, household)

        # Contact tracing for the households infected by the household currently traced
        child_households_not_traced = [h for h in household.spread_to() if not h.isolated]
        for child in child_households_not_traced:
            self.attempt_contact_trace_of_household(child, household)


class IncrementContactTracingUK(IncrementContactTracingBehaviour):

    def increment_contact_tracing(self, time: int):

        # TODO update the below - going to hospital is not included in the model
        """
        Performs a days worth of contact tracing by:
        * Looking for nodes that have been admitted to hospital. Once a node is admitted to hospital, it's house is isolated
        * Looks for houses where the contact tracing delay is over and moves them to the contact traced state
        * Looks for houses in the contact traced state, and checks them for symptoms. If any of them have symptoms, the house is isolated

        The isolation function also assigns contact tracing times to any houses that had contact with that household

        For each node that is contact traced
        """

        # Isolate all households under observation that now display symptoms (excludes those who will not take up isolation if prob <1)
        self.receive_pcr_test_results()

        [
            self._contact_tracing.contact_trace_household.isolate_household(node.household(), time)
            for node in self._network.all_nodes()
            if node.symptom_onset_time <= time
               and node.received_positive_test_result
               and not node.isolated
               and not node.completed_isolation
        ]

        [
            self.propagate_contact_tracing(node)
            for node in self._network.all_nodes()
            if node.received_result and not node.propagated_contact_tracing
        ]

    def propagate_contact_tracing(self, node: 'Node', time: int):
        """
        To be called after a node in a household either reports their symptoms, and gets tested, when a household that
        is under surveillance develops symptoms + gets tested.
        """
        # update the propagation data
        node.propagated_contact_tracing = True
        node.time_propagated_tracing = time

        # Contact tracing attempted for the household that infected the household currently propagating the infection
        infected_by_node = node.infected_by_node()

        # If the node was globally infected, we are backwards tracing and the infecting node is not None
        if not node.locally_infected() and infected_by_node:

            # if the infector is not already isolated and the time the node was infected captured by going backwards
            # the node.time_infected is when they had a contact with their infector.
            if  not infected_by_node.isolated and node.time_infected >= node.symptom_onset_time - \
                    self.number_of_days_to_trace_backwards:

                # Then attempt to contact trace the household of the node that infected you
                self.attempt_contact_trace_of_household(
                    house_to=infected_by_node.household(),
                    house_from=node.household(),
                    days_since_contact_occurred=time - node.time_infected
                    )

        # spread_to_global_node_time_tuples stores a list of tuples, where the first element is the node_id
        # of a node who was globally infected by the node, and the second element is the time of transmission
        for global_infection in node.spread_to_global_node_time_tuples:

            # Get the child node_id and the time of transmission/time of contact
            child_node_id, time_t = global_infection

            child_node = self._network.node(child_node_id)

            # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
            if time_t >= node.symptom_onset_time - self.number_of_days_to_trace_backwards and \
                    time_t <= node.symptom_onset_time + self.number_of_days_to_trace_forwards and \
                    not child_node.isolated:

                self.attempt_contact_trace_of_household(
                    house_to=child_node.household(),
                    house_from=node.household(),
                    days_since_contact_occurred=time - time_t
                    )

    def receive_pcr_test_results(self, time: int):
        """For nodes who would receive a PCR test result today, update
        """
        # self reporting infections
        [
            self.pcr_test_node(node, time)
            for node in self._network.all_nodes()
            if node.time_of_reporting + node.testing_delay == time
            and not node.received_result
            and not node.contact_traced
        ]

        # contact traced nodes
        [
            self.pcr_test_node(node, time)
            for node in self._network.all_nodes()
            if node.symptom_onset_time + node.testing_delay == time
            and node.contact_traced
            and not node.received_result
        ]

    def pcr_test_node(self, node: 'Node', time: int):
        """Given the nodes infectious age, will that node test positive

        Args:
            node (Node): The node to be tested today
            time (int): Current time in days
        """
        node.received_result = True

        infectious_age_when_tested = time - node.testing_delay - node.time_infected

        prob_positive_result = self.prob_testing_positive_pcr_func(infectious_age_when_tested)

        if npr.binomial(1, prob_positive_result) == 1:
            node.received_positive_test_result = True
        else:
            node.received_positive_test_result = False


class IncrementContactTracingContactModelTest(IncrementContactTracingBehaviour):
    def __init__(self, network: Network, contact_tracing: ContactTracing,
                 lfa_testing_requires_confirmatory_pcr,
                 lfa_tested_nodes_book_pcr_on_symptom_onset,
                 prob_testing_positive_pcr_func):

        super(IncrementContactTracingContactModelTest, self).__init__(network, contact_tracing)
        self.LFA_testing_requires_confirmatory_PCR = lfa_testing_requires_confirmatory_pcr
        self.lfa_tested_nodes_book_pcr_on_symptom_onset = lfa_tested_nodes_book_pcr_on_symptom_onset
        self.prob_testing_positive_pcr_func = prob_testing_positive_pcr_func


    def increment_contact_tracing(self, time: int):
        [
            self.propagate_contact_tracing(node, time)
            for node in self._network.all_nodes()
            if node.received_positive_test_result
               and node.avenue_of_testing == 'PCR'
               and not node.propagated_contact_tracing
        ]

        if not self.LFA_testing_requires_confirmatory_PCR:
            [
                self.propagate_contact_tracing(node, time)
                for node in self._network.all_nodes()
                if node.received_positive_test_result
                   and node.avenue_of_testing == 'LFA'
                   and not node.propagated_contact_tracing
            ]

        elif self.LFA_testing_requires_confirmatory_PCR:
            [
                self.propagate_contact_tracing(node, time)
                for node in self._network.all_nodes()
                if node.confirmatory_PCR_test_result_time == time
                   and node.confirmatory_PCR_result_was_positive
                   and node.avenue_of_testing == 'LFA'
                   and not node.propagated_contact_tracing
            ]

    def propagate_contact_tracing(self, node: 'NodeContactModel', time: int):
        """
        To be called after a node in a household either reports their symptoms, and gets tested, when a household that is under surveillance develops symptoms + gets tested.
        """

        # TODO: Refactor this monster
        # There are really 3 contact tracing algorithms going on here
        # 1) Trace on non-confirmatory PCR result
        # 2) Trace on confirmatory PCR result

        # update the propagation data
        node.propagated_contact_tracing = True
        node.time_propagated_tracing = time

        # Contact tracing attempted for the household that infected the household currently propagating the infection
        infected_by_node = node.infected_by_node()

        # If the node was globally infected, we are backwards tracing and the infecting node is not None
        if not node.locally_infected() and infected_by_node:

            # if the infector is not already isolated and the time the node was infected captured by going backwards
            # the node.time_infected is when they had a contact with their infector.
            if node.avenue_of_testing == 'PCR':

                if not infected_by_node.isolated and node.time_infected >= node.symptom_onset_time - self.number_of_days_to_trace_backwards:

                    # Then attempt to contact trace the household of the node that infected you
                    self.attempt_contact_trace_of_household(
                        house_to=infected_by_node.household(),
                        house_from=node.household(),
                        days_since_contact_occurred=time - node.time_infected
                        )

            elif node.avenue_of_testing == 'LFA':

                if not self.LFA_testing_requires_confirmatory_PCR:

                    if not infected_by_node.isolated and node.time_infected >= node.positive_test_time - self.number_of_days_prior_to_LFA_result_to_trace:

                        # Then attempt to contact trace the household of the node that infected you
                        self.attempt_contact_trace_of_household(
                            house_to=infected_by_node.household(),
                            house_from=node.household(),
                            days_since_contact_occurred=time - node.time_infected
                            )

        # spread_to_global_node_time_tuples stores a list of tuples, where the first element is the node_id
        # of a node who was globally infected by the node, and the second element is the time of transmission
        for global_infection in node.spread_to_global_node_time_tuples:

            # Get the child node_id and the time of transmission/time of contact
            child_node_id, time_t = global_infection

            child_node = self.network.node(child_node_id)

            if node.avenue_of_testing == 'PCR':

                # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
                if time_t >= node.symptom_onset_time - self.number_of_days_to_trace_backwards and \
                        time_t <= node.symptom_onset_time + self.number_of_days_to_trace_forwards and \
                        not child_node.isolated:

                    self.attempt_contact_trace_of_household(
                        house_to=child_node.household(),
                        house_from=node.household(),
                        days_since_contact_occurred=time - time_t
                        )

            elif node.avenue_of_testing == 'LFA':

                if not self.LFA_testing_requires_confirmatory_PCR:

                    # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
                    if time_t >= node.positive_test_time - self.number_of_days_prior_to_LFA_result_to_trace:

                        self.attempt_contact_trace_of_household(
                            house_to=child_node.household(),
                            house_from=node.household(),
                            days_since_contact_occurred=time - time_t
                            )

    def receive_pcr_test_results(self, time: int):
        """
        For nodes who would receive a PCR test result today, update
        """

        if self.lfa_tested_nodes_book_pcr_on_symptom_onset:

            # self reporting infections who have not been contact traced
            [
                self.pcr_test_node(node, time)
                for node in self._network.all_nodes()
                if node.time_of_reporting + node.testing_delay == time
                and not node.received_result
                and not node.contact_traced
            ]

            # contact traced nodes should book a pcr test if they develop symptoms
            # we assume that this occurs at symptom onset time since they are traced
            # and on the lookout for developing symptoms
            [
                self.pcr_test_node(node, time)
                for node in self._network.all_nodes()
                if node.symptom_onset_time + node.testing_delay == time
                and not node.received_result
                and node.contact_traced
            ]

        else:

            [
                self.pcr_test_node(node, time)
                for node in self._network.all_nodes()
                if node.time_of_reporting + node.testing_delay == time
                and not node.received_result
                and not node.contact_traced
                and not node.being_lateral_flow_tested
            ]

    def pcr_test_node(self, node: 'NodeContactModel', time: int):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (NodeContactModel): The node to be tested today
            time (int): Current time in days
        """
        node.received_result = True

        infectious_age_when_tested = time - node.testing_delay - node.time_infected

        prob_positive_result = self.prob_testing_positive_pcr_func(infectious_age_when_tested)

        if npr.binomial(1, prob_positive_result) == 1:
            node.received_positive_test_result = True
            node.avenue_of_testing = 'PCR'
            node.positive_test_time = time
        else:
            node.received_positive_test_result = False
            node.avenue_of_testing = 'PCR'
