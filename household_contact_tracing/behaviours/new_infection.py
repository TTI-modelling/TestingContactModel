from __future__ import annotations
import sys
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import numpy
import numpy as np

from household_contact_tracing.network.contact_tracing_network import ContactTracingNetwork, \
    Household, ContactTracingNode
from household_contact_tracing.parameterised import Parameterised


class NewInfection(ABC, Parameterised):
    """
        An abstract base class used to represent the highest level 'new infection' behaviour.

        Note:   This class forms part of a 'Strategy' pattern. All child classes implement a family of possible
                behaviours or strategies (ways of adding a new infection).
                Add further child classes to add new behaviour types (strategies) that can be selected and updated at
                design or run-time.

        Attributes
        ----------
        network: ContactTracingNetwork
            The store of Nodes and households used in the simulation
        symptom_reporting_delay (int todo?)
            Delay in symptom reporting
        Todo: descriptions of each attribute
        incubation_period_delay
        asymptomatic_prob
        infection_reporting_prob
        test_delay
        test_before_propagate_tracing
        prob_has_trace_app
        propensity_imperfect_quarantine
        node_prob_will_take_up_lfa_testing
        propensity_risky_behaviour_lfa_testing
        proportion_with_propensity_miss_lfa_tests
        node_will_uptake_isolation_prob


        Methods
        -------

        new_infection(self, time: int, household: Household, infecting_node: Optional[Node] = None)
            Add a new infected Node to the model.
            :param time: The current simulation time.
            :param household: The Household to create the new infection in.
            :param infecting_node: The source of the new infection.

    """

    def __init__(self, network: ContactTracingNetwork, params: dict):
        self.network = network
        self.symptom_reporting_delay = 1
        self.incubation_period_delay = 5
        self.asymptomatic_prob = 0.5
        self.infection_reporting_prob = 1.0
        self.test_delay = 1
        self.test_before_propagate_tracing = True
        self.prob_has_trace_app = 0
        self.propensity_imperfect_quarantine = 0
        self.node_prob_will_take_up_lfa_testing = 1
        self.propensity_risky_behaviour_lfa_testing = 0
        self.proportion_with_propensity_miss_lfa_tests = 0.
        self.node_will_uptake_isolation_prob = 1

        self.update_params(params)

    @abstractmethod
    def new_infection(self, time: int, household: Household, infecting_node: Optional[ContactTracingNode] = None):
        """Add a new infected Node to the model.
        :param time: The current simulation time.
        :param household: The Household to create the new infection in.
        :param infecting_node: The source of the new infection.
        """

    def is_asymptomatic_infection(self) -> bool:
        """Determine whether a node"""
        return numpy.random.binomial(1, self.asymptomatic_prob) == 1

    def incubation_period(self, asymptomatic: bool) -> int:
        if asymptomatic:
            return sys.maxsize
        else:
            return round(self.incubation_period_delay)

    def reporting_delay(self, asymptomatic: bool) -> int:
        if asymptomatic:
            return sys.maxsize
        else:
            return round(self.symptom_reporting_delay)

    def testing_delay(self) -> int:
        if self.test_before_propagate_tracing is False:
            return 0
        else:
            return round(self.test_delay)

    def has_contact_tracing_app(self) -> bool:
        return numpy.random.binomial(1, self.prob_has_trace_app) == 1

    def get_propensity_imperfect_isolation(self) -> bool:
        return np.random.choice([True, False], p=(self.propensity_imperfect_quarantine, 1 -
                                                  self.propensity_imperfect_quarantine))

    def will_take_up_lfa_testing(self) -> bool:
        return np.random.binomial(1, self.node_prob_will_take_up_lfa_testing) == 1

    def will_engage_in_risky_behaviour_while_being_lfa_tested(self):
        """Will the node engage in more risky behaviour if they are being LFA tested?
        """
        if np.random.binomial(1, self.propensity_risky_behaviour_lfa_testing) == 1:
            return True
        else:
            return False

    def propensity_to_miss_lfa_tests(self) -> bool:
        return np.random.binomial(1, self.proportion_with_propensity_miss_lfa_tests) == 1

    def will_uptake_isolation(self) -> bool:
        """Based on the node_will_uptake_isolation_prob, return a bool
        where True implies they do take up isolation and False implies they do not uptake isolation

        Returns:
            bool: If True they uptake isolation, if False they do not uptake isolation
        """
        return numpy.random.choice([True, False], p=(self.node_will_uptake_isolation_prob, 1 -
                                                     self.node_will_uptake_isolation_prob))


class NewInfectionHouseholdLevel(NewInfection):

    def new_infection(self, time: int, household: Household, infecting_node: Optional[ContactTracingNode] = None):
        """Add a new infected Node to the model."""
        asymptomatic = self.is_asymptomatic_infection()

        # Symptom onset time
        symptom_onset_time = time + self.incubation_period(asymptomatic)

        # If the node is asymptomatic, we need to generate a pseudo symptom onset time
        if asymptomatic:
            pseudo_symptom_onset_time = self.incubation_period(asymptomatic=False)
        else:
            pseudo_symptom_onset_time = symptom_onset_time

        # When a node reports its infection
        if not asymptomatic and np.random.binomial(1, self.infection_reporting_prob) == 1:
            will_report_infection = True
            time_of_reporting = symptom_onset_time + self.reporting_delay(asymptomatic)
        else:
            will_report_infection = False
            time_of_reporting = float('Inf')

        # We assign each node a recovery period of 21 days, after 21 days the probability of
        # causing a new infections is 0, due to the generation time distribution
        recovery_time = time + 14

        # If the household has the propensity to use the contact tracing app, decide
        # if the node uses the app.
        if household.propensity_trace_app:
            has_trace_app = self.has_contact_tracing_app()
        else:
            has_trace_app = False

        isolation_uptake = self.will_uptake_isolation()

        if household.isolated and isolation_uptake:
            node_is_isolated = True
        else:
            node_is_isolated = False

        new_node = self.network.add_node(time_infected=time,
                                         household_id=household.id, isolated=node_is_isolated,
                                         will_uptake_isolation=isolation_uptake,
                                         propensity_imperfect_isolation=self.get_propensity_imperfect_isolation(),
                                         asymptomatic=asymptomatic, contact_traced=household.contact_traced,
                                         symptom_onset_time=symptom_onset_time,
                                         pseudo_symptom_onset_time=pseudo_symptom_onset_time,
                                         recovery_time=recovery_time,
                                         will_report_infection=will_report_infection,
                                         time_of_reporting=time_of_reporting,
                                         has_contact_tracing_app=has_trace_app,
                                         testing_delay=self.testing_delay(),
                                         infecting_node=infecting_node)

        # Each house now stores the ID's of which nodes are stored inside the house,
        # so that quarantining can be done at the household level
        household.nodes.append(new_node)


class NewInfectionIndividualTracingDailyTesting(NewInfection):

    def new_infection(self, time: int, household: Household, infecting_node: Optional[ContactTracingNode] = None):
        """Add a new infection to the model and network. Attributes are randomly generated.

        This method passes additional attribute, relevant to the lateral flow testing.

        Args:
            time: The current simulation time.
            household: The household that the node is being added to
            infecting_node: The infecting node
        """

        node_will_take_up_lfa_testing = self.will_take_up_lfa_testing()

        if household.being_lateral_flow_tested:

            time_started_lfa_testing = household.being_lateral_flow_tested_start_time

            if node_will_take_up_lfa_testing:
                node_being_lateral_flow_tested = True

            else:
                node_being_lateral_flow_tested = False

        else:
            node_being_lateral_flow_tested = False
            time_started_lfa_testing = float('Inf')

        additional_attributes = {
            'being_lateral_flow_tested': node_being_lateral_flow_tested,
            'time_started_lfa_testing': time_started_lfa_testing,
            'received_positive_test_result': False,
            'received_result': None,
            'avenue_of_testing': None,
            'positive_test_time': None,
            'node_will_take_up_lfa_testing': node_will_take_up_lfa_testing,
            'confirmatory_PCR_result_was_positive': None,
            'taken_confirmatory_PCR_test': False,
            'confirmatory_PCR_test_time': None,
            'confirmatory_PCR_test_result_time': None,
            'propensity_risky_behaviour_lfa_testing':
                self.will_engage_in_risky_behaviour_while_being_lfa_tested(),
            'propensity_to_miss_lfa_tests': self.propensity_to_miss_lfa_tests()
        }

        asymptomatic = self.is_asymptomatic_infection()

        # Symptom onset time
        symptom_onset_time = time + self.incubation_period(asymptomatic)

        # If the node is asymptomatic, we need to generate a pseudo symptom onset time
        if asymptomatic:
            pseudo_symptom_onset_time = self.incubation_period(asymptomatic=False)
        else:
            pseudo_symptom_onset_time = symptom_onset_time

        # When a node reports its infection
        if not asymptomatic and np.random.binomial(1, self.infection_reporting_prob) == 1:
            will_report_infection = True
            time_of_reporting = symptom_onset_time + self.reporting_delay(asymptomatic)
        else:
            will_report_infection = False
            time_of_reporting = float('Inf')

        # We assign each node a recovery period of 21 days, after 21 days the probability of
        # causing a new infections is 0, due to the generation time distribution
        recovery_time = time + 14

        # If the household has the propensity to use the contact tracing app, decide
        # if the node uses the app.
        if household.propensity_trace_app:
            has_trace_app = self.has_contact_tracing_app()
        else:
            has_trace_app = False

        isolation_uptake = self.will_uptake_isolation()

        if household.isolated and isolation_uptake:
            node_is_isolated = True
        else:
            node_is_isolated = False

        new_node = self.network.add_node(time_infected=time,
                                         household_id=household.id,
                                         isolated=node_is_isolated,
                                         will_uptake_isolation=isolation_uptake,
                                         propensity_imperfect_isolation=self.get_propensity_imperfect_isolation(),
                                         asymptomatic=asymptomatic,
                                         contact_traced=household.contact_traced,
                                         symptom_onset_time=symptom_onset_time,
                                         pseudo_symptom_onset_time=pseudo_symptom_onset_time,
                                         recovery_time=recovery_time,
                                         will_report_infection=will_report_infection,
                                         time_of_reporting=time_of_reporting,
                                         has_contact_tracing_app=has_trace_app,
                                         testing_delay=self.testing_delay(),
                                         additional_attributes=additional_attributes,
                                         infecting_node=infecting_node)

        # Each house now stores the ID's of which nodes are stored inside the house,
        # so that quarantining can be done at the household level
        household.nodes.append(new_node)
