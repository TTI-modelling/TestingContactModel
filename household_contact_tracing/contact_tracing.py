from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy.random as npr
from collections.abc import Callable

from household_contact_tracing.network import Network, Household, EdgeType, Node, TestType, InfectionStatus

if TYPE_CHECKING:
    import household_contact_tracing.behaviours.isolation as isolation
    import household_contact_tracing.behaviours.pcr_testing as pcr
    import household_contact_tracing.behaviours.contact_trace_household as tracing


class ContactTracing:
    """ 'Context' class for contact tracing processes/strategies (Strategy pattern) """

    def __init__(self, network: Network,
                 contact_trace_household: tracing.ContactTraceHouseholdBehaviour,
                 increment: IncrementContactTracingBehaviour,
                 update_isolation: isolation.UpdateIsolationBehaviour,
                 pcr_testing: Optional[pcr.PCRTestingBehaviour], params: dict):
        self._network = network

        # Declare behaviours
        self.contact_trace_household_behaviour = contact_trace_household
        self.increment_behaviour = increment
        self.update_isolation_behaviour = update_isolation
        self.pcr_testing_behaviour = pcr_testing

        # Parameter Inputs:
        # contact tracing parameters
        self.contact_tracing_success_prob = 0.5
        self.do_2_step = False
        self.contact_trace_delay = 1
        self.policy_for_household_contacts_of_a_positive_case = 'lfa testing no quarantine'
        self.LFA_testing_requires_confirmatory_PCR = False
        self.node_daily_prob_lfa_test = 1
        self.lfa_tested_nodes_book_pcr_on_symptom_onset = True
        self.number_of_days_to_trace_backwards = 2
        self.number_of_days_to_trace_forwards = 7
        self.recall_probability_fall_off = 1
        self.number_of_days_prior_to_LFA_result_to_trace: int = 2

        # isolation or quarantine parameters
        self.self_isolation_duration = 7
        self.quarantine_duration = 14
        self.lateral_flow_testing_duration = 7

        # contact tracing functions (runtime updatable)
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
    def update_isolation_behaviour(self) -> isolation.UpdateIsolationBehaviour:
        return self._update_isolation_behaviour

    @update_isolation_behaviour.setter
    def update_isolation_behaviour(self, update_isolation_behaviour: isolation.UpdateIsolationBehaviour):
        self._update_isolation_behaviour = update_isolation_behaviour
        if self._update_isolation_behaviour:
            self._update_isolation_behaviour.contact_tracing = self

    @property
    def contact_trace_household_behaviour(self) -> tracing.ContactTraceHouseholdBehaviour:
        return self._contact_trace_household_behaviour

    @contact_trace_household_behaviour.setter
    def contact_trace_household_behaviour(self, contact_trace_household_behaviour: tracing.ContactTraceHouseholdBehaviour):
        self._contact_trace_household_behaviour = contact_trace_household_behaviour
        if self._contact_trace_household_behaviour:
            self._contact_trace_household_behaviour.contact_tracing = self

    @property
    def increment_behaviour(self) -> IncrementContactTracingBehaviour:
        return self._increment_behaviour

    @increment_behaviour.setter
    def increment_behaviour(self, increment_behaviour: IncrementContactTracingBehaviour):
        self._increment_behaviour = increment_behaviour
        if self._increment_behaviour:
            self._increment_behaviour.contact_tracing = self

    @property
    def pcr_testing_behaviour(self) -> pcr.PCRTestingBehaviour:
        return self._pcr_testing_behaviour

    @pcr_testing_behaviour.setter
    def pcr_testing_behaviour(self, pcr_testing_behaviour: pcr.PCRTestingBehaviour):
        self._pcr_testing_behaviour = pcr_testing_behaviour
        if self._pcr_testing_behaviour:
            self._pcr_testing_behaviour.contact_tracing = self

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

    def receive_pcr_test_results(self, time: int):
        if self.pcr_testing_behaviour:
            self.pcr_testing_behaviour.receive_pcr_test_results(time)

    def pcr_test_node(self, node: Node, time: int):
        if self.pcr_testing_behaviour:
            self.pcr_testing_behaviour.pcr_test_node(node, time)

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
        node.confirmatory_PCR_test_result_time = time + node.testing_delay
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

    def isolate_self_reporting_cases(self, time):
        """Applies the isolation status to nodes who have reached their self-report time.
        They may of course decide to not adhere to said isolation, or may be a member of a household
        who will not uptake isolation
        """
        for node in self.network.all_nodes():
            if node.will_uptake_isolation:
                 if node.time_of_reporting == time:
                    node.isolated = True

    def release_nodes_from_quarantine_or_isolation(self, time):
        """If a node has completed the quarantine according to the following rules, they are released from
        quarantine.

        You are released from isolation if:
            * it has been 10 days since your symptoms onset
            and
            * it has been a minimum of 14 days since your household was isolated
        """

        # We consider two distinct cases, and define logic for each case
        self.release_nodes_who_completed_isolation(time)
        self.release_nodes_who_completed_quarantine(time)

    def release_nodes_who_completed_quarantine(self, time):
        """If a node is currently in quarantine, and has completed the quarantine period then we release them from
        quarantine.

        An individual is in quarantine if they have been contact traced, and have not had symptom onset.

        A quarantined individual is released from quarantine if it has been quarantine_duration since they last had
        contact with a known case.
        In our model, this corresponds to the time of infection.
        """
        for node in self.network.all_nodes():
            # For nodes who do not self-report, and are in the same household as their infector
            # (if they do not self-report they will not isolate; if contact traced, they will be quarantining for the
            # quarantine duration)
            # if node.household_id == node.infected_by_node().household_id:
            if node.infected_by_node():
                if (node.infection_status(time) == InfectionStatus.unknown_infection) & node.isolated:
                    if node.locally_infected():

                        if time >= (node.household().earliest_recognised_symptom_onset(model_time=time)
                                    + self.quarantine_duration):
                            node.isolated = False
                            node.completed_isolation = True
                            node.completed_isolation_reason = 'completed_quarantine'
                            node.completed_isolation_time = time
                    # For nodes who do not self-report, and are not in the same household as their infector
                    # (if they do not self-report they will not isolate; if contact traced, they will be quarantining for
                    # the quarantine duration)
                    elif node.contact_traced & (time >= node.time_infected + self.quarantine_duration):
                        node.isolated = False
                        node.completed_isolation = True
                        node.completed_isolation_time = time
                        node.completed_isolation_reason = 'completed_quarantine'

    def release_nodes_who_completed_isolation(self, time):
        """
        Nodes leave self-isolation, rather than quarantine, when their infection status is either known (ie tested) or
        when they are in a         contact traced household and they develop symptoms (they might then go on to get a
        test, but they isolate regardless).
        Nodes in contact traced households do not have a will_report_infection probability: if they develop symptoms,
        they are a self-recognised infection who might or might not go on to test and become a known infection.

        If it has been isolation_duration since these individuals have had symptom onset, then they are released
        from isolation.
        """
        for node in self.network.all_nodes():
            if node.isolated:
                infection_status = node.infection_status(time)
                if infection_status in [InfectionStatus.known_infection,
                                        infection_status.self_recognised_infection]:
                    if time >= node.symptom_onset_time + self.self_isolation_duration:
                        node.isolated = False
                        node.completed_isolation = True
                        node.completed_isolation_time = time
                        node.completed_isolation_reason = 'completed_isolation'

    def release_nodes_from_lateral_flow_testing_or_isolation(self, time: int):
        """If a node has completed the quarantine according to the following rules, they are
        released from quarantine.

        You are released from isolation if:
            * it has been 10 days since your symptoms onset (Changed from 7 to reflect updated policy, Nov 2020)
        You are released form lateral flow testing if you have reached the end of the lateral flow testing period
        and not yet been removed because you are positive

        """

        # We consider two distinct cases, and define logic for each case
        self.release_nodes_who_completed_isolation(time)
        self.release_nodes_who_completed_lateral_flow_testing(time)

    def release_nodes_who_completed_isolation(self, time: int):
        """
        Nodes leave self-isolation, rather than quarantine, when their infection status is either known (ie tested) or
        when they are in a contact traced household and they develop symptoms (they might then go on to get a test, but
        they isolate regardless). Nodes in contact traced households do not have a will_report_infection probability:
        if they develop symptoms, they are a self-recognised infection who might or might not go on to test and become
        a known infection.

        If it has been isolation_duration since these individuals have had symptom onset, then they are released
        from isolation.
        """
        for node in self.network.all_nodes():
            if node.isolated:
                infection_status = node.infection_status(time)
                if infection_status in [InfectionStatus.known_infection,
                                        InfectionStatus.self_recognised_infection]:
                    if node.avenue_of_testing == TestType.lfa:
                        if time >= node.positive_test_time + self.self_isolation_duration:
                            node.isolated = False
                            node.completed_isolation = True
                            node.completed_isolation_time = time
                            node.completed_isolation_reason = 'completed_isolation'
                    else:
                        if time >= node.symptom_onset_time + self.self_isolation_duration:
                            # this won't include nodes who tested positive due to LF tests who do not have symptoms
                            node.isolated = False
                            node.completed_isolation = True
                            node.completed_isolation_time = time
                            node.completed_isolation_reason = 'completed_isolation'


    def release_nodes_who_completed_lateral_flow_testing(self, time: int):
        """If a node is currently in lateral flow testing, and has completed this period then we release them from
        testing.

        An individual is in lateral flow testing if they have been contact traced, and have not had symptom onset.

        They continue to be lateral flow tested until the duration of this period is up OR they test positive on
        lateral flow and they are isolated and traced.

        A lateral flow tested individual is released from testing if it has been 'lateral_flow_testing_duration' since
        they last had contact with a known case.
        In our model, this corresponds to the time of infection.
        """

        for node in self.network.all_nodes():
            if time >= node.time_started_lfa_testing + self.lateral_flow_testing_duration \
                    and node.being_lateral_flow_tested:
                node.being_lateral_flow_tested = False
                node.completed_lateral_flow_testing_time = time

        # for node in self.network.all_nodes():

        #     # For nodes who do not self-report, and are in the same household as their infector
        #     # (if they do not self-report they will not isolate; if contact traced, they will be lateral flow testing
        #     for the lateral_flow_testing_duration unless they test positive)
        #     #if node.household_id == node.infected_by_node().household_id:
        #     if node.infected_by_node():
        #         #if (node.infection_status(self.time) == "unknown_infection") & node.being_lateral_flow_tested:
        #         if node.being_lateral_flow_tested:
        #             if node.locally_infected():

        #                 if self.time >=
        #                 (node.household().earliest_recognised_symptom_onset_or_lateral_flow_test(model_time =
        #                 self.time) + self.lateral_flow_testing_duration):
        #                     node.being_lateral_flow_tested = False
        #                     node.completed_lateral_flow_testing_time = self.time

        #         # For nodes who do not self-report, and are not in the same household as their infector
        #         # (if they do not self-report they will not isolate; if contact traced, they will be lateral flow
        #         testing for the lateral_flow_testing_duration unless they test positive)
        #             elif node.contact_traced & (self.time >= node.time_infected + self.lateral_flow_testing_duration):
        #                 node.being_lateral_flow_tested = False
        #                 node.completed_lateral_flow_testing_time = self.time


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


class IncrementContactTracingHouseholdLevel(IncrementContactTracingBehaviour):

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
            self.contact_tracing.contact_trace_household_behaviour.isolate_household(node.household(), time)
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


class IncrementContactTracingIndividualLevel(IncrementContactTracingHouseholdLevel):

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
        self.contact_tracing.receive_pcr_test_results(time)

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
                    self.contact_tracing.number_of_days_to_trace_backwards:

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
            if time_t >= node.symptom_onset_time - self.contact_tracing.number_of_days_to_trace_backwards and \
                    time_t <= node.symptom_onset_time + self.contact_tracing.number_of_days_to_trace_forwards and \
                    not child_node.isolated:

                self.attempt_contact_trace_of_household(
                    house_to=child_node.household(),
                    house_from=node.household(),
                    days_since_contact_occurred=time - time_t,
                    time=time
                    )

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
                           self.contact_tracing.recall_probability_fall_off ** days_since_contact_occurred

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


class IncrementContactTracingIndividualDailyTesting(IncrementContactTracingIndividualLevel):

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

        elif self.contact_tracing.LFA_testing_requires_confirmatory_PCR:
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
                        node.time_infected >= node.symptom_onset_time - \
                            self.contact_tracing.number_of_days_to_trace_backwards:

                    # Then attempt to contact trace the household of the node that infected you
                    self.attempt_contact_trace_of_household(
                        house_to=infected_by_node.household(),
                        house_from=node.household(),
                        days_since_contact_occurred=time - node.time_infected,
                        time=time)

            elif node.avenue_of_testing == TestType.lfa:

                if not self.contact_tracing.LFA_testing_requires_confirmatory_PCR:

                    if not infected_by_node.isolated and node.time_infected >= \
                            node.positive_test_time - self.contact_tracing.number_of_days_prior_to_LFA_result_to_trace:

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
                if time_t >= node.symptom_onset_time - self.contact_tracing.number_of_days_to_trace_backwards and \
                        time_t <= node.symptom_onset_time + self.contact_tracing.number_of_days_to_trace_forwards and \
                        not child_node.isolated:

                    self.attempt_contact_trace_of_household(
                        house_to=child_node.household(),
                        house_from=node.household(),
                        days_since_contact_occurred=time - time_t,
                        time=time)

            elif node.avenue_of_testing == TestType.lfa:

                if not self.contact_tracing.LFA_testing_requires_confirmatory_PCR:

                    # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
                    if time_t >= node.positive_test_time - \
                            self.contact_tracing.number_of_days_prior_to_LFA_result_to_trace:

                        self.attempt_contact_trace_of_household(
                            house_to=child_node.household(),
                            house_from=node.household(),
                            days_since_contact_occurred=time - time_t,
                            time=time)
