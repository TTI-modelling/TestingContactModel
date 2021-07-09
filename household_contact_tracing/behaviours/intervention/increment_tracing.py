from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from household_contact_tracing.network import Network, Node, Household, EdgeType, TestType
from household_contact_tracing.parameterised import Parameterised


class IncrementTracing(ABC, Parameterised):
    """
        An abstract base class used to represent the highest level 'increment tracing' behaviour.
        Implements contact tracing increments.

        Note:   This class forms part of a 'Strategy' pattern. All child classes implement a family of possible
                behaviours or strategies (ways of incrementing contact tracing).
                Add further child classes to add new behaviour types (strategies) that can be selected and updated at
                design or run-time.

        Attributes
        ----------
        network: Network
            The store of Nodes and households used in the simulation

        Methods
        -------

        increment_contact_tracing(self, time: int)
            Performs a days worth of contact tracing.

    """

    def __init__(self, network: Network, params: dict):
        self.network = network
        self.do_2_step = False
        self.contact_tracing_success_prob = 0.5
        self.contact_trace_delay = 1
        self.number_of_days_to_trace_backwards = 2
        self.number_of_days_to_trace_forwards = 7
        self.recall_probability_fall_off = 1
        self.number_of_days_prior_to_LFA_result_to_trace: int = 2
        self.lfa_tested_nodes_book_pcr_on_symptom_onset = True
        self.LFA_testing_requires_confirmatory_PCR = False

        self.update_params(params)

    @abstractmethod
    def increment_contact_tracing(self, time: int):
        """Performs a days worth of contact tracing."""


class IncrementTracingHouseholdLevel(IncrementTracing):

    def increment_contact_tracing(self, time: int):
        """
        Performs a days worth of contact tracing by:
        * Looks for houses where the contact tracing delay is over and moves them to the contact traced state
        * Looks for houses in the contact traced state, and checks them for symptoms. If any of them have symptoms,
        the house is isolated

        The intervention function also assigns contact tracing times to any houses that had contact with that household

        For each node that is contact traced
        """

        # Isolate all households under observation that now display symptoms (excludes those
        # who will not take up intervention if prob <1)
        for node in self.network.all_nodes():
            if node.symptom_onset_time <= time:
                if node.contact_traced:
                    if not node.isolated:
                        if not node.completed_isolation:
                            node.household.isolate_household(time)

        # Propagate the contact tracing for all households that self-reported and have had their
        # test results come back
        for node in self.network.all_nodes():
            if node.time_of_reporting + node.testing_delay == time:
                if not node.household.propagated_contact_tracing:
                    self.propagate_contact_tracing(node.household, time)

        # Propagate the contact tracing for all households that are isolated due to exposure,
        # have developed symptoms and had a test come back
        for node in self.network.all_nodes():
            if node.symptom_onset_time <= time:
                if not node.household.propagated_contact_tracing:
                    if node.household.isolated_time + node.testing_delay <= time:
                        self.propagate_contact_tracing(node.household, time)

        # Update the contact tracing index of households
        # That is, re-evaluate how far away they are from a known infected case
        # (traced + symptom_onset_time + testing_delay)
        self.update_contact_tracing_index(time)

        if self.do_2_step:
            # Propagate the contact tracing from any households with a contact tracing index of 1
            for household in self.network.all_households:
                if household.contact_tracing_index == 1:
                    if not household.propagated_contact_tracing:
                        if household.isolated:
                            self.propagate_contact_tracing(household, time)

    def propagate_contact_tracing(self, household: Household, time: int):
        """
        To be called after a node in a household either reports their symptoms, and gets tested,
        when a household that is under surveillance develops symptoms + gets tested.
        """
        # update the propagation data
        household.propagated_contact_tracing = True

        # Contact tracing attempted for the household that infected the household currently
        # propagating the infection
        # If infected by = None, then it is the origin node, a special case
        if household.infected_by and not household.infected_by.isolated:
            self.attempt_contact_trace_of_household(household.infected_by, household, time)

        # Contact tracing for the households infected by the household currently traced
        child_households_not_traced = [h for h in household.spread_to if not h.isolated]
        for child in child_households_not_traced:
            self.attempt_contact_trace_of_household(child, household, time)

    def attempt_contact_trace_of_household(self, house_to: Household, house_from: Household,
                                           time: int, contact_trace_delay: int = 0):
        # Decide if the edge was traced by the app
        app_traced = self.network.is_edge_app_traced(self.network.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self.contact_tracing_success_prob

        # is the trace successful
        if np.random.binomial(1, success_prob) == 1:
            # Update the list of traced households from this one
            house_from.contact_traced_households.append(house_to)

            # Assign the household a contact tracing index, 1 more than its parent tracer
            house_to.contact_tracing_index = house_from.contact_tracing_index + 1

            # work out the time delay
            contact_trace_delay = contact_trace_delay + self.contact_trace_delay
            proposed_time_until_contact_trace = time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed
            # time until contact trace
            # Note this starts as infinity
            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < house_to.time_until_contact_traced:
                house_to.time_until_contact_traced = proposed_time_until_contact_trace
                house_to.being_contact_traced_from = house_from

            # Edge labelling
            if app_traced:
                self.network.label_edges_between_houses(house_to, house_from, EdgeType.app_traced)
            else:
                self.network.label_edges_between_houses(house_to, house_from, EdgeType.between_house)
        else:
            self.network.label_edges_between_houses(house_to, house_from, EdgeType.failed_contact_tracing)

    def update_contact_tracing_index(self, time: int):
        for household in self.network.all_households:
            # loop over households with non-zero indexes, those that have been contact traced but
            # with
            if household.contact_tracing_index != 0:
                for node in household.nodes:

                    # Necessary conditions for an index 1 household to propagate tracing:
                    # The node must have onset of symptoms
                    # The node households must be isolated
                    # The testing delay must be passed
                    # The testing delay starts when the house have been isolated and symptoms have
                    # onset
                    critical_time = max(node.symptom_onset_time, household.isolated_time)

                    if critical_time + node.testing_delay <= time:
                        household.contact_tracing_index = 0

                        for index_1_hh in household.contact_traced_households:
                            if index_1_hh.contact_tracing_index == 2:
                                index_1_hh.contact_tracing_index = 1


class IncrementTracingIndividualLevel(IncrementTracingHouseholdLevel):

    def __init__(self, network: Network, params: dict):

        super().__init__(network, params)
        self.prob_pcr_positive = self.default_prob_pcr_positive

    @property
    def prob_pcr_positive(self) -> Callable[[int], float]:
        return self._prob_pcr_positive

    @prob_pcr_positive.setter
    def prob_pcr_positive(self, fn: Callable[[int], float]):
        self._prob_pcr_positive = fn

    @staticmethod
    def default_prob_pcr_positive(infectious_age):
        """Default PCR test result probability."""
        if infectious_age in [4, 5, 6]:
            return 0
        else:
            return 0

    def pcr_test_node(self, node: Node, time: int, prob_pcr_positive: Callable):
        node.received_result = True
        infectious_age_when_tested = time - node.testing_delay - node.time_infected
        prob_positive_result = prob_pcr_positive(infectious_age_when_tested)
        node.avenue_of_testing = TestType.pcr

        if np.random.binomial(1, prob_positive_result) == 1:
            node.received_positive_test_result = True
            node.positive_test_time = time
        else:
            node.received_positive_test_result = False

    def receive_pcr_test_results(self, time: int, prob_pcr_positive: Callable):
        # self reporting infections
        for node in self.network.all_nodes():
            if node.time_of_reporting + node.testing_delay == time:
                if not node.contact_traced:
                    if not node.received_result:
                        self.pcr_test_node(node, time, prob_pcr_positive)

        # contact traced nodes
        for node in self.network.all_nodes():
            if node.symptom_onset_time + node.testing_delay == time:
                if node.contact_traced:
                    if not node.received_result:
                        self.pcr_test_node(node, time, prob_pcr_positive)

    def increment_contact_tracing(self, time: int):

        # TODO update the below - going to hospital is not included in the model
        """
        Performs a days worth of contact tracing by:
        * Looking for nodes that have been admitted to hospital. Once a node is admitted to hospital, its house is
        isolated
        * Looks for houses where the contact tracing delay is over and moves them to the contact traced state
        * Looks for houses in the contact traced state, and checks them for symptoms. If any of them have symptoms, the
        house is isolated

        The intervention function also assigns contact tracing times to any houses that had contact with that household

        For each node that is contact traced
        """

        # Isolate all households under observation that now display symptoms (excludes those who will not take up
        # intervention if prob <1)
        self.receive_pcr_test_results(time, self.prob_pcr_positive)

        for node in self.network.all_nodes():
            if node.symptom_onset_time <= time:
                if node.received_positive_test_result:
                    if not node.isolated:
                        if not node.completed_isolation:
                            node.household.isolate_household(time)

        for node in self.network.all_nodes():
            if node.received_result:
                if not node.propagated_contact_tracing:
                    self.propagate_contact_tracing(node, time)

    def propagate_contact_tracing(self, node: Node, time: int):
        """
        To be called after a node in a household either reports their symptoms, and gets tested,
        when a household that is under surveillance develops symptoms + gets tested.
        """
        # update the propagation data
        node.propagated_contact_tracing = True

        # Contact tracing attempted for the household that infected the household currently
        # propagating the infection
        infected_by_node = node.infecting_node

        # If the node was globally infected, we are backwards tracing and the infecting node is
        # not None
        if not node.locally_infected() and infected_by_node:

            # if the infector is not already isolated and the time the node was infected captured
            # by going backwards
            # the node.time_infected is when they had a contact with their infector.
            if not infected_by_node.isolated and node.time_infected >= node.symptom_onset_time - \
                    self.number_of_days_to_trace_backwards:

                # Then attempt to contact trace the household of the node that infected you
                self.attempt_contact_trace_of_household(
                    house_to=infected_by_node.household,
                    house_from=node.household,
                    time=time,
                    days_since_contact_occurred=time - node.time_infected
                    )

        # spread_to_global_node_time_tuples stores a list of tuples, where the first element is
        # the node_id of a node who was globally infected by the node, and the second element is
        # the time of transmission
        for global_infection in node.spread_to_global_node_time_tuples:

            # Get the child node_id and the time of transmission/time of contact
            child_node_id, time_t = global_infection

            child_node = self.network.node(child_node_id)

            # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
            if time_t >= node.symptom_onset_time - self.number_of_days_to_trace_backwards and \
                    time_t <= node.symptom_onset_time + self.number_of_days_to_trace_forwards and \
                    not child_node.isolated:

                self.attempt_contact_trace_of_household(
                    house_to=child_node.household,
                    house_from=node.household,
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
        app_traced = self.network.is_edge_app_traced(self.network.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self.contact_tracing_success_prob * \
                           self.recall_probability_fall_off ** days_since_contact_occurred

        # is the trace successful
        if np.random.binomial(1, success_prob) == 1:
            # Update the list of traced households from this one
            house_from.contact_traced_households.append(house_to)

            # Assign the household a contact tracing index, 1 more than its parent tracer
            house_to.contact_tracing_index = house_from.contact_tracing_index + 1

            # work out the time delay
            proposed_time_until_contact_trace = time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed time
            # until contact trace. Note this starts as infinity.
            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < house_to.time_until_contact_traced:
                house_to.time_until_contact_traced = proposed_time_until_contact_trace
                house_to.being_contact_traced_from = house_from

            # Edge labelling
            if app_traced:
                self.network.label_edges_between_houses(house_to, house_from, EdgeType.app_traced)
            else:
                self.network.label_edges_between_houses(house_to, house_from, EdgeType.between_house)
        else:
            self.network.label_edges_between_houses(house_to, house_from, EdgeType.failed_contact_tracing)


class IncrementTracingIndividualDailyTesting(IncrementTracingIndividualLevel):

    def receive_pcr_test_results(self, time: int, prob_pcr_positive: Callable):
        """For nodes who would receive a PCR test result today, update"""

        if self.lfa_tested_nodes_book_pcr_on_symptom_onset:
            super().receive_pcr_test_results(time, prob_pcr_positive)
        else:
            for node in self.network.all_nodes():
                if node.time_of_reporting + node.testing_delay == time:
                    if not node.contact_traced:
                        if not node.received_result:
                            if not node.being_lateral_flow_tested:
                                self.pcr_test_node(node, time, prob_pcr_positive)

    def increment_contact_tracing(self, time: int):
        for node in self.network.all_nodes():
            if node.received_positive_test_result:
                if node.avenue_of_testing == TestType.pcr:
                    if not node.propagated_contact_tracing:
                        self.propagate_contact_tracing(node, time)

        if not self.LFA_testing_requires_confirmatory_PCR:
            for node in self.network.all_nodes():
                if node.received_positive_test_result:
                    if node.avenue_of_testing == TestType.lfa:
                        if not node.propagated_contact_tracing:
                            self.propagate_contact_tracing(node, time)

        elif self.LFA_testing_requires_confirmatory_PCR:
            for node in self.network.all_nodes():
                if node.confirmatory_PCR_test_result_time == time:
                    if node.confirmatory_PCR_result_was_positive:
                        if node.avenue_of_testing == TestType.lfa:
                            if not node.propagated_contact_tracing:
                                self.propagate_contact_tracing(node, time)

    def propagate_contact_tracing(self, node: Node, time: int):
        """
        To be called after a node in a household either reports their symptoms, and gets tested,
        when a household that is under surveillance develops symptoms + gets tested.
        """

        # There are 2 contact tracing algorithms going on here
        # 1) Trace on non-confirmatory PCR result
        # 2) Trace on confirmatory PCR result

        # update the propagation data
        node.propagated_contact_tracing = True

        # Contact tracing attempted for the household that infected the household currently
        # propagating the infection
        infected_by_node = node.infecting_node

        # If the node was globally infected, we are backwards tracing and the infecting node
        # is not None
        if not node.locally_infected() and infected_by_node:

            # if the infector is not already isolated and the time the node was infected captured
            # by going backwards the node.time_infected is when they had a contact with their
            # infector.
            if node.avenue_of_testing == TestType.pcr:

                if not infected_by_node.isolated and \
                        node.time_infected >= node.symptom_onset_time - \
                            self.number_of_days_to_trace_backwards:

                    # Then attempt to contact trace the household of the node that infected you
                    self.attempt_contact_trace_of_household(
                        house_to=infected_by_node.household,
                        house_from=node.household,
                        days_since_contact_occurred=time - node.time_infected,
                        time=time)

            elif node.avenue_of_testing == TestType.lfa:

                if not self.LFA_testing_requires_confirmatory_PCR:

                    if not infected_by_node.isolated and node.time_infected >= \
                            node.positive_test_time - self.number_of_days_prior_to_LFA_result_to_trace:

                        # Then attempt to contact trace the household of the node that infected you
                        self.attempt_contact_trace_of_household(
                            house_to=infected_by_node.household,
                            house_from=node.household,
                            days_since_contact_occurred=time - node.time_infected,
                            time=time )

        # spread_to_global_node_time_tuples stores a list of tuples, where the first element is
        # the node_id of a node who was globally infected by the node, and the second element is
        # the time of transmission
        for global_infection in node.spread_to_global_node_time_tuples:

            # Get the child node_id and the time of transmission/time of contact
            child_node_id, time_t = global_infection

            child_node = self.network.node(child_node_id)

            if node.avenue_of_testing == TestType.pcr:

                # If the node was infected 2 days prior to symptom onset, or 7 days post and is
                # not already isolated
                if time_t >= node.symptom_onset_time - self.number_of_days_to_trace_backwards:
                        if time_t <= node.symptom_onset_time + self.number_of_days_to_trace_forwards:
                            if not child_node.isolated:

                                self.attempt_contact_trace_of_household(
                                    house_to=child_node.household,
                                    house_from=node.household,
                                    days_since_contact_occurred=time - time_t,
                                    time=time)

            elif node.avenue_of_testing == TestType.lfa:

                if not self.LFA_testing_requires_confirmatory_PCR:

                    # If the node was infected 2 days prior to symptom onset, or 7 days post and
                    # is not already isolated
                    if time_t >= node.positive_test_time - \
                            self.number_of_days_prior_to_LFA_result_to_trace:

                        self.attempt_contact_trace_of_household(
                            house_to=child_node.household,
                            house_from=node.household,
                            days_since_contact_occurred=time - time_t,
                            time=time)