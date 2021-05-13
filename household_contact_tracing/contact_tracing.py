from __future__ import annotations
import numpy.random as npr
from collections.abc import Callable

from household_contact_tracing.network import Network, Household, EdgeType, Node, TestType


class ContactTracing:
    """ 'Context' class for contact tracing processes/strategies (Strategy pattern) """

    def __init__(self, network: Network, contact_trace_household: ContactTraceHouseholdBehaviour,
                 increment: IncrementContactTracingBehaviour, update_isolation: UpdateIsolationBehaviour, params: dict):
        self._network = network

        # Declare behaviours
        self.contact_trace_household_behaviour = contact_trace_household
        self.increment_behaviour = increment
        self.update_isolation_behaviour = update_isolation

        # Parameter Inputs:
        # contact tracing parameters
        self.contact_tracing_success_prob = 0.5
        self.do_2_step = False
        self.hh_propensity_to_use_trace_app = 1
        self.test_before_propagate_tracing = True
        self.test_delay = 1
        self.contact_trace_delay = 1
        self.policy_for_household_contacts_of_a_positive_case = 'lfa testing no quarantine'
        self.LFA_testing_requires_confirmatory_PCR = False
        self.node_daily_prob_lfa_test = 1
        self.prob_testing_positive_lfa_func = self.prob_testing_positive_lfa
        self.prob_testing_positive_pcr_func = self.prob_testing_positive_pcr

        self.current_LFA_positive_nodes = []

        # Update instance variables with anything in params
        for param_name in self.__dict__:
            if param_name in params:
                self.__dict__[param_name] = params[param_name]

    @property
    def network(self) -> Network:
        return self._network

    @property
    def update_isolation_behaviour(self) -> UpdateIsolationBehaviour:
        return self._update_isolation_behaviour

    @update_isolation_behaviour.setter
    def update_isolation_behaviour(self, update_isolation_behaviour: UpdateIsolationBehaviour):
        self._update_isolation_behaviour = update_isolation_behaviour
        self._update_isolation_behaviour.contact_tracing = self

    @property
    def contact_trace_household_behaviour(self) -> ContactTraceHouseholdBehaviour:
        return self._contact_trace_household_behaviour

    @contact_trace_household_behaviour.setter
    def contact_trace_household_behaviour(self, contact_trace_household_behaviour: ContactTraceHouseholdBehaviour):
        self._contact_trace_household_behaviour = contact_trace_household_behaviour
        self._contact_trace_household_behaviour.contact_tracing = self

    @property
    def increment_behaviour(self) -> IncrementContactTracingBehaviour:
        return self._increment_behaviour

    @increment_behaviour.setter
    def increment_behaviour(self, increment_behaviour: IncrementContactTracingBehaviour):
        self._increment_behaviour = increment_behaviour
        self._increment_behaviour.contact_tracing = self

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

    def prob_testing_positive_pcr(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0
        else:
            return 0

    def prob_testing_positive_lfa(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 1
        else:
            return 0

    def update_isolation(self, time: int):
        if self.update_isolation_behaviour:
            self.update_isolation_behaviour.update_isolation(time)

    def contact_trace_household(self, household: Household, time: int):
        if self.contact_trace_household_behaviour:
            self.contact_trace_household_behaviour.contact_trace_household(household, time)

    def increment(self, time: int):
        if self.increment_behaviour:
            self.increment_behaviour.increment_contact_tracing(time)

    def hh_propensity_use_trace_app(self) -> bool:
        if npr.binomial(1, self.hh_propensity_to_use_trace_app) == 1:
            return True
        else:
            return False

    def testing_delay(self) -> int:
        if self.test_before_propagate_tracing is False:
            return 0
        else:
            return round(self.test_delay)

    def apply_policy_for_household_contacts_of_a_positive_case(self, household: Household, time: int):
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
            self.contact_trace_household_behaviour.isolate_household(household, time)
        else:
            raise Exception("""policy_for_household_contacts_of_a_positive_case not recognised. Must be one of the 
            following options:
                * "lfa testing no quarantine"
                * "lfa testing and quarantine"
                * "no lfa testing only quarantine" """)

    def start_lateral_flow_testing_household(self, household: Household, time: int):
        """Sets the household to the lateral flow testing status so that new within household infections are tested

        All nodes in the household start lateral flow testing

        Args:
            household (Household): The household which is initiating testing
        """

        household.being_lateral_flow_tested = True
        household.being_lateral_flow_tested_start_time = time

        for node in household.nodes():
            if node.node_will_take_up_lfa_testing and not node.received_positive_test_result and \
                    not node.being_lateral_flow_tested:
                node.being_lateral_flow_tested = True
                node.time_started_lfa_testing = time

    def start_lateral_flow_testing_household_and_quarantine(self, household: Household, time: int):
        """Sets the household to the lateral flow testing status so that new within household infections are tested

        All nodes in the household start lateral flow testing and start quarantining

        Args:
            household (Household): The household which is initiating testing
        """
        household.being_lateral_flow_tested = True
        household.being_lateral_flow_tested_start_time = time
        household.isolated = True
        household.isolated_time = True
        household.contact_traced = True

        for node in household.nodes():
            if node.node_will_take_up_lfa_testing and not node.received_positive_test_result and \
                    not node.being_lateral_flow_tested:
                node.being_lateral_flow_tested = True
                node.time_started_lfa_testing = time

            if node.will_uptake_isolation:
                node.isolated = True

    def lfa_test_node(self, node: Node, time: int):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (Node): The node to be tested today
        """

        infectious_age = time - node.time_infected

        prob_positive_result = self.prob_testing_positive_lfa_func(infectious_age)

        if npr.binomial(1, prob_positive_result) == 1:
            return True
        else:
            return False

    def will_lfa_test_today(self, node: Node) -> bool:

        if node.propensity_to_miss_lfa_tests:

            if npr.binomial(1, self.node_daily_prob_lfa_test) == 1:
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

        [
            self.apply_policy_for_household_contacts_of_a_positive_case(node.household(), time)
            for node in self.network.all_nodes()
            if node.confirmatory_PCR_test_result_time == time
        ]

    def get_positive_lateral_flow_nodes(self, time: int):
        """Performs a days worth of lateral flow testing.

        Returns:
            List[Nodes]: A list of nodes who have tested positive through the lateral flow tests.
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

            if not node.household().applied_policy_for_household_contacts_of_a_positive_case and \
                    not self.LFA_testing_requires_confirmatory_PCR:
                self.apply_policy_for_household_contacts_of_a_positive_case(node.household(), time)


    def take_confirmatory_pcr_test(self, node: Node, time: int):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (Node): The node to be tested today
        """

        infectious_age_when_tested = time - node.time_infected
        prob_positive_result = self.prob_testing_positive_pcr_func(infectious_age_when_tested)

        node.confirmatory_PCR_test_time = time
        node.confirmatory_PCR_test_result_time = self.time + node.testing_delay
        node.taken_confirmatory_PCR_test = True

        if npr.binomial(1, prob_positive_result) == 1:
            node.confirmatory_PCR_result_was_positive = True

        else:
            node.confirmatory_PCR_result_was_positive = False


    def confirmatory_pcr_test_LFA_nodes(self):
        """Nodes who receive a positive LFA result will be tested using a PCR test.
        """
        for node in self.current_LFA_positive_nodes:
            if not node.taken_confirmatory_PCR_test:
                self.take_confirmatory_pcr_test(node)

    def act_on_positive_LFA_tests(self, time: int):
        """For nodes who test positive on their LFA test, take the appropriate action depending on the policy
        """
        self.current_LFA_positive_nodes = self.get_positive_lateral_flow_nodes(time)

        self.isolate_positive_lateral_flow_tests(time)

        if self.LFA_testing_requires_confirmatory_PCR:
            self.confirmatory_pcr_test_LFA_nodes()


#Todo - Peter: for Network?  All UpdateIsolation Behaviours below and sub-classes???
class UpdateIsolationBehaviour:
    def __init__(self, network: Network):
        self._network = network
        self._contact_tracing = None

    @property
    def contact_tracing(self) -> ContactTracing:
        return self._contact_tracing

    @contact_tracing.setter
    def contact_tracing(self, contact_tracing: ContactTracing):
        self._contact_tracing = contact_tracing

    def update_isolation(self, time):
        pass

    def update_all_households_contact_traced(self, time):
        # Update the contact traced status for all households that have had the contact tracing process get there
        [
            self._contact_tracing.contact_trace_household(household, time)
            for household in self._network.houses.all_households()
            if household.time_until_contact_traced <= time
               and not household.contact_traced
        ]


class UpdateIsolationHousehold(UpdateIsolationBehaviour):

    def update_isolation(self, time):
        # Update the contact traced status for all households that have had the contact tracing process get there
        self.update_all_households_contact_traced(time)

        # Isolate all non isolated households where the infection has been reported (excludes those who will not
        # take up isolation if prob <1)
        [
            self._contact_tracing.contact_trace_household_behaviour.isolate_household(node.household(), time)
            for node in self._network.all_nodes()
            if node.time_of_reporting + node.testing_delay == time
            and not node.household().isolated
            and not node.household().contact_traced
        ]


class UpdateIsolationUK(UpdateIsolationBehaviour):
    def update_isolation(self, time):
        # Update the contact traced status for all households that have had the contact
        # tracing process get there
        self.update_all_households_contact_traced(time)

        # Isolate all non isolated households where the infection has been reported
        # (excludes those who will not take up isolation if prob <1)
        [
            self._contact_tracing.contact_trace_household_behaviour.isolate_household(node.household(), time)
            for node in self._network.all_nodes()
            if node.time_of_reporting + node.testing_delay == time
               and node.received_positive_test_result
               and not node.household().isolated
               and not node.household().contact_traced
        ]


class UpdateIsolationContactModelTest(UpdateIsolationBehaviour):
    def update_isolation(self, time):
        # Update the contact traced status for all households that have had the contact tracing process get there
        self.update_all_households_contact_traced(time)

        # Isolate all non isolated households where the infection has been reported (excludes those who will not
        # take up isolation if prob <1)
        new_pcr_test_results = [
            node for node in self._network.all_nodes()
            if node.positive_test_time == time
            and node.avenue_of_testing == TestType.PCR
            and node.received_positive_test_result
        ]

        [
            self.apply_policy_for_household_contacts_of_a_positive_case(node.household())
            for node in new_pcr_test_results
            if not node.household().applied_policy_for_household_contacts_of_a_positive_case
        ]


#Todo - Peter: for Network?  All ContactTraceHousehold Behaviours below and sub-classes???
class ContactTraceHouseholdBehaviour:

    def __init__(self, network: Network):
        self._network = network

    def contact_trace_household(self, household: Household, time: int):
        pass

    def label_node_edges_between_houses(self, house_to: Household, house_from: Household, new_edge_type):
        pass

    #Todo - Peter: for Network?
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

    # Todo - Peter: for Network?
    def quarantine_traced_node(self, household):
        traced_node = self.find_traced_node(household)

        # the traced node should go into quarantine
        if not traced_node.isolated and traced_node.will_uptake_isolation:
            #Todo: AG: Peter/Martyn checking 2nd operand is traced_node and not node (as before)
            traced_node.isolated = True

    # Todo - Peter: for Network?
    def find_traced_node(self, household):
        # work out which was the traced node
        tracing_household = self._network.houses.household(household.being_contact_traced_from)
        traced_node_id = self._network.get_edge_between_household(household, tracing_household)[0]
        return self._network.node(traced_node_id)

    # Todo - Peter: for Network?
    def isolate_household_if_symptomatic_nodes(self, household: Household, time: int):
        symptomatic_nodes = [node for node in household.nodes() if
                             node.symptom_onset_time <= time and not node.completed_isolation]
        if symptomatic_nodes:
            self.isolate_household(household, time)

    # Todo - Peter: for Network?
    def isolate_household(self, household: Household, time: int):
        """
        Isolates a house so that all infectives in that household may no longer infect others.

        If the house is being surveillance due to a successful contact trace, and not due to reporting symptoms,
        update the edge label to display this.

        For households that were connected to this household, they are assigned a time until contact traced

        When a house has been contact traced, all nodes in the house are under surveillance for symptoms. When a node
        becomes symptomatic, the house moves to isolation status.
        """

        # Makes sure the isolate household is never applied multiple times to the same household
        if not household.isolated:

            # update the household and all nodes in the household to the contact traced status
            household.contact_traced = True
            for node in household.nodes():
                node.contact_traced = True

            # Households have a probability to take up isolation if traced

            # The house moves to isolated status if it has been assigned to take up isolation if trace, given a
            # probability
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

                # Initially the edge is assigned the contact tracing label, may be updated if the contact tracing
                # does not succeed
                if self._network.is_edge_app_traced(
                        self._network.get_edge_between_household(household, house_which_contact_traced)):
                    self.label_node_edges_between_houses(household, house_which_contact_traced,
                                                         EdgeType.app_traced.name)
                else:
                    self.label_node_edges_between_houses(household, house_which_contact_traced,
                                                         EdgeType.between_house.name)

                    # We update the label of every edge so that we can tell which household have been contact traced
                    # when we visualise
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
    def __init__(self, network: Network):
        self._network = network
        self._contact_tracing = None

    @property
    def contact_tracing(self) -> ContactTracing:
        return self._contact_tracing

    @contact_tracing.setter
    def contact_tracing(self, contact_tracing: ContactTracing):
        self._contact_tracing = contact_tracing

    def increment_contact_tracing(self, time: int):
        pass

    def receive_pcr_test_results(self, time: int):
        pass


class IncrementContactTracingHousehold(IncrementContactTracingBehaviour):

    def increment_contact_tracing(self, time: int):
        """
        Performs a days worth of contact tracing by:
        * Looks for houses where the contact tracing delay is over and moves them to the contact traced state
        * Looks for houses in the contact traced state, and checks them for symptoms. If any of them have symptoms,
        the house is isolated

        The isolation function also assigns contact tracing times to any houses that had contact with that household

        For each node that is contact traced
        """

        # Isolate all households under observation that now display symptoms (excludes those who will not take up
        # isolation if prob <1)
        [
            self._contact_tracing.contact_trace_household_behaviour.isolate_household(node.household(), time)
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

        # Propagate the contact tracing for all households that are isolated due to exposure, have developed symptoms
        # and had a test come back
        [
            self.propagate_contact_tracing(node.household(), time)
            for node in self._network.all_nodes()
            if node.symptom_onset_time <= time
            and not node.household().propagated_contact_tracing
            and node.household().isolated_time + node.testing_delay <= time
        ]

        # Update the contact tracing index of households
        # That is, re-evaluate how far away they are from a known infected case
        # (traced + symptom_onset_time + testing_delay)
        self.update_contact_tracing_index(time)

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
        To be called after a node in a household either reports their symptoms, and gets tested, when a household
        that is under surveillance develops symptoms + gets tested.
        """
        # update the propagation data
        household.propagated_contact_tracing = True
        household.time_propagated_tracing = time

        # Contact tracing attempted for the household that infected the household currently propagating the infection

        infected_by = household.infected_by()

        # If infected by = None, then it is the origin node, a special case
        if infected_by and not infected_by.isolated:
            self.attempt_contact_trace_of_household(infected_by, household, time)

        # Contact tracing for the households infected by the household currently traced
        child_households_not_traced = [h for h in household.spread_to() if not h.isolated]
        for child in child_households_not_traced:
            self.attempt_contact_trace_of_household(child, household, time)

    def attempt_contact_trace_of_household(self,
                                           house_to: Household,
                                           house_from: Household,
                                           time: int,
                                           contact_trace_delay: int = 0):
        # Decide if the edge was traced by the app
        app_traced = self._network.is_edge_app_traced(self._network.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self._contact_tracing.contact_tracing_success_prob

        # is the trace successful
        if (npr.binomial(1, success_prob) == 1):
            # Update the list of traced households from this one
            house_from.contact_traced_household_ids.append(house_to.house_id)

            # Assign the household a contact tracing index, 1 more than its parent tracer
            house_to.contact_tracing_index = house_from.contact_tracing_index + 1

            # work out the time delay
            contact_trace_delay = contact_trace_delay + self._contact_tracing.contact_trace_delay
            proposed_time_until_contact_trace = time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed time until contact trace
            # Note this starts as infinity
            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < house_to.time_until_contact_traced:
                house_to.time_until_contact_traced = proposed_time_until_contact_trace
                house_to.being_contact_traced_from = house_from.house_id

            # Edge labelling
            if app_traced:
                self._contact_tracing.contact_trace_household_behaviour.label_node_edges_between_houses(
                    house_to, house_from, EdgeType.app_traced.name)
            else:
                self._contact_tracing.contact_trace_household_behaviour.label_node_edges_between_houses(
                    house_to, house_from, EdgeType.between_house.name)
        else:
            self._contact_tracing.contact_trace_household_behaviour.label_node_edges_between_houses(
                house_to, house_from, EdgeType.failed_contact_tracing.name)

    # Todo - Peter: for Network?
    def update_contact_tracing_index(self, time):
        for household in self._network.houses.all_households():
            # loop over households with non-zero indexes, those that have been contact traced but with
            if household.contact_tracing_index != 0:
                for node in household.nodes():

                    # Necessary conditions for an index 1 household to propagate tracing:
                    # The node must have onset of symptoms
                    # The node households must be isolated
                    # The testing delay must be passed
                    # The testing delay starts when the house have been isolated and symptoms have onset
                    critical_time = max(node.symptom_onset_time, household.isolated_time)

                    if critical_time + node.testing_delay <= time:
                        household.contact_tracing_index = 0

                        for index_1_hh in household.contact_traced_households():
                            if index_1_hh.contact_tracing_index == 2:
                                index_1_hh.contact_tracing_index = 1


class IncrementContactTracingUK(IncrementContactTracingHousehold):

    def __init__(self, network,
                 number_of_days_to_trace_backwards,
                 number_of_days_to_trace_forwards,
                 recall_probability_fall_off
    ):
        super(IncrementContactTracingUK, self).__init__(network)
        self.number_of_days_to_trace_backwards = number_of_days_to_trace_backwards
        self.number_of_days_to_trace_forwards = number_of_days_to_trace_forwards
        self.recall_probability_fall_off = recall_probability_fall_off

    def increment_contact_tracing(self, time: int):

        # TODO update the below - going to hospital is not included in the model
        """
        Performs a days worth of contact tracing by:
        * Looking for nodes that have been admitted to hospital. Once a node is admitted to hospital, its house is
        isolated
        * Looks for houses where the contact tracing delay is over and moves them to the contact traced state
        * Looks for houses in the contact traced state, and checks them for symptoms. If any of them have symptoms, the
        house is isolated

        The isolation function also assigns contact tracing times to any houses that had contact with that household

        For each node that is contact traced
        """

        # Isolate all households under observation that now display symptoms (excludes those who will not take up
        # isolation if prob <1)
        self.receive_pcr_test_results(time)

        [
            self._contact_tracing.contact_trace_household_behaviour.isolate_household(node.household(), time)
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

    def propagate_contact_tracing(self, node: Node, time: int):
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
                    time=time,
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
                    days_since_contact_occurred=time - time_t,
                    time=time
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

    def pcr_test_node(self, node: Node, time: int):
        """Given the nodes infectious age, will that node test positive

        Args:
            node (Node): The node to be tested today
            time (int): Current time in days
        """
        node.received_result = True

        infectious_age_when_tested = time - node.testing_delay - node.time_infected

        prob_positive_result = self.contact_tracing.prob_testing_positive_pcr_func(infectious_age_when_tested)

        if npr.binomial(1, prob_positive_result) == 1:
            node.received_positive_test_result = True
        else:
            node.received_positive_test_result = False

    def attempt_contact_trace_of_household(self,
                                           house_to: Household,
                                           house_from: Household,
                                           days_since_contact_occurred: int,
                                           time: int,
                                           contact_trace_delay: int = 0):
        # Decide if the edge was traced by the app
        app_traced = self._network.is_edge_app_traced(self._network.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self.contact_tracing.contact_tracing_success_prob * \
                           self.recall_probability_fall_off ** days_since_contact_occurred

        # is the trace successful
        if (npr.binomial(1, success_prob) == 1):
            # Update the list of traced households from this one
            house_from.contact_traced_household_ids.append(house_to.house_id)

            # Assign the household a contact tracing index, 1 more than its parent tracer
            house_to.contact_tracing_index = house_from.contact_tracing_index + 1

            # work out the time delay
            proposed_time_until_contact_trace = time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed time until contact trace
            # Note this starts as infinity
            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < house_to.time_until_contact_traced:
                house_to.time_until_contact_traced = proposed_time_until_contact_trace
                house_to.being_contact_traced_from = house_from.house_id

            # Edge labelling
            if app_traced:
                self._contact_tracing.contact_trace_household_behaviour.label_node_edges_between_houses(
                    house_to, house_from, EdgeType.app_traced.name)
            else:
                self._contact_tracing.contact_trace_household_behaviour.label_node_edges_between_houses(
                    house_to, house_from, EdgeType.between_house.name)
        else:
            self._contact_tracing.contact_trace_household_behaviour.label_node_edges_between_houses(
                house_to, house_from, EdgeType.failed_contact_tracing.name)


class IncrementContactTracingContactModelTest(IncrementContactTracingUK):
    def __init__(self, network: Network,
                 lfa_tested_nodes_book_pcr_on_symptom_onset,
                 number_of_days_to_trace_backwards,
                 number_of_days_to_trace_forwards,
                 recall_probability_fall_off,
                 number_of_days_prior_to_LFA_result_to_trace):
        super(IncrementContactTracingContactModelTest, self).__init__(network,
                                                                      number_of_days_to_trace_backwards,
                                                                      number_of_days_to_trace_forwards,
                                                                      recall_probability_fall_off)
        self.lfa_tested_nodes_book_pcr_on_symptom_onset = lfa_tested_nodes_book_pcr_on_symptom_onset
        self.number_of_days_prior_to_LFA_result_to_trace = number_of_days_prior_to_LFA_result_to_trace

    def increment_contact_tracing(self, time: int):
        [
            self.propagate_contact_tracing(node, time)
            for node in self._network.all_nodes()
            if node.received_positive_test_result
               and node.avenue_of_testing == TestType.pcr
               and not node.propagated_contact_tracing
        ]

        if not self.contact_tracing.LFA_testing_requires_confirmatory_PCR:
            [
                self.propagate_contact_tracing(node, time)
                for node in self._network.all_nodes()
                if node.received_positive_test_result
                   and node.avenue_of_testing == TestType.lfa
                   and not node.propagated_contact_tracing
            ]

        elif self.LFA_testing_requires_confirmatory_PCR:
            [
                self.propagate_contact_tracing(node, time)
                for node in self._network.all_nodes()
                if node.confirmatory_PCR_test_result_time == time
                   and node.confirmatory_PCR_result_was_positive
                   and node.avenue_of_testing == TestType.lfa
                   and not node.propagated_contact_tracing
            ]

    def propagate_contact_tracing(self, node: Node, time: int):
        """
        To be called after a node in a household either reports their symptoms, and gets tested, when a household
        that is under surveillance develops symptoms + gets tested.
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
            if node.avenue_of_testing == TestType.pcr:

                if not infected_by_node.isolated and \
                        node.time_infected >= node.symptom_onset_time - self.number_of_days_to_trace_backwards:

                    # Then attempt to contact trace the household of the node that infected you
                    self.attempt_contact_trace_of_household(
                        house_to=infected_by_node.household(),
                        house_from=node.household(),
                        days_since_contact_occurred=time - node.time_infected,
                        time=time)

            elif node.avenue_of_testing == TestType.lfa:

                if not self.LFA_testing_requires_confirmatory_PCR:

                    if not infected_by_node.isolated and node.time_infected >= \
                            node.positive_test_time - self.number_of_days_prior_to_LFA_result_to_trace:

                        # Then attempt to contact trace the household of the node that infected you
                        self.attempt_contact_trace_of_household(
                            house_to=infected_by_node.household(),
                            house_from=node.household(),
                            days_since_contact_occurred=time - node.time_infected,
                            time=time )

        # spread_to_global_node_time_tuples stores a list of tuples, where the first element is the node_id
        # of a node who was globally infected by the node, and the second element is the time of transmission
        for global_infection in node.spread_to_global_node_time_tuples:

            # Get the child node_id and the time of transmission/time of contact
            child_node_id, time_t = global_infection

            child_node = self._network.node(child_node_id)

            if node.avenue_of_testing == TestType.pcr:

                # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
                if time_t >= node.symptom_onset_time - self.number_of_days_to_trace_backwards and \
                        time_t <= node.symptom_onset_time + self.number_of_days_to_trace_forwards and \
                        not child_node.isolated:

                    self.attempt_contact_trace_of_household(
                        house_to=child_node.household(),
                        house_from=node.household(),
                        days_since_contact_occurred=time - time_t,
                        time=time)

            elif node.avenue_of_testing == TestType.lfa:

                if not self.LFA_testing_requires_confirmatory_PCR:

                    # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
                    if time_t >= node.positive_test_time - self.number_of_days_prior_to_LFA_result_to_trace:

                        self.attempt_contact_trace_of_household(
                            house_to=child_node.household(),
                            house_from=node.household(),
                            days_since_contact_occurred=time - time_t,
                            time=time)

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

    def pcr_test_node(self, node: Node, time: int):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (NodeContactModel): The node to be tested today
            time (int): Current time in days
        """
        node.received_result = True

        infectious_age_when_tested = time - node.testing_delay - node.time_infected

        prob_positive_result = self.contact_tracing.prob_testing_positive_pcr_func(infectious_age_when_tested)

        node.avenue_of_testing = TestType.pcr

        if npr.binomial(1, prob_positive_result) == 1:
            node.received_positive_test_result = True
            node.positive_test_time = time
        else:
            node.received_positive_test_result = False
