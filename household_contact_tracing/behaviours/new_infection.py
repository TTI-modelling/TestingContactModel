from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from household_contact_tracing.network import Network, Node
    from household_contact_tracing.infection import Infection


class NewInfection:
    def __init__(self, network: Network):
        self._network = network
        self._infection = None

    @property
    def infection(self) -> Infection:
        return self._infection

    @infection.setter
    def infection(self, infection: Infection):
        self._infection = infection

    def new_infection(self,
                      time: int,
                      node_count: int,
                      household_id: int,
                      serial_interval=None,
                      infecting_node: Optional[Node] = None,
                      additional_attributes: Optional[dict] = None):
        pass


class NewInfectionHouseholdLevel(NewInfection):

    def new_infection(self,
                      time: int,
                      node_count: int,
                      household_id: int,
                      serial_interval=None,
                      infecting_node: Optional[Node] = None,
                      additional_attributes: Optional[dict] = None):
        """
        Adds a new infection to the graph along with the following attributes:
        t - when they were infected
        offspring - how many offspring they produce

        Inputs::
        G - the network object
        time - the time when the new infection happens
        node_count - how many nodes are currently in the network
        """
        asymptomatic = self._infection.is_asymptomatic_infection()

        # Symptom onset time
        symptom_onset_time = time + self._infection.incubation_period(asymptomatic)

        # If the node is asymptomatic, we need to generate a pseudo symptom onset time
        if asymptomatic:
            pseudo_symptom_onset_time = self._infection.incubation_period(asymptomatic=False)
        else:
            pseudo_symptom_onset_time = symptom_onset_time

        # When a node reports its infection
        if not asymptomatic and np.random.binomial(1, self._infection.infection_reporting_prob) == 1:
            will_report_infection = True
            time_of_reporting = symptom_onset_time + self._infection.reporting_delay(asymptomatic)
        else:
            will_report_infection = False
            time_of_reporting = float('Inf')

        # We assign each node a recovery period of 21 days, after 21 days the probability of
        # causing a new infections is 0, due to the generation time distribution
        recovery_time = time + 14

        household = self._network.houses.household(household_id)

        # If the household has the propensity to use the contact tracing app, decide
        # if the node uses the app.
        if household.propensity_trace_app:
            has_trace_app = self._infection.has_contact_tracing_app()
        else:
            has_trace_app = False

        # in case you want to add non-default additional attributes
        default_additional_attributes = {}

        if additional_attributes:
            default_additional_attributes = {**default_additional_attributes,
                                             **additional_attributes}

        isolation_uptake = self._infection.will_uptake_isolation()

        if household.isolated and isolation_uptake:
            node_is_isolated = True
        else:
            node_is_isolated = False

        new_node = self._network.add_node(node_id=node_count, time=time,
                                          household_id=household_id, isolated=node_is_isolated,
                               will_uptake_isolation=isolation_uptake,
                               propensity_imperfect_isolation=self._infection.get_propensity_imperfect_isolation(),
                               asymptomatic=asymptomatic, contact_traced=household.contact_traced,
                               symptom_onset_time=symptom_onset_time,
                               pseudo_symptom_onset_time=pseudo_symptom_onset_time,
                               serial_interval=serial_interval, recovery_time=recovery_time,
                               will_report_infection=will_report_infection,
                               time_of_reporting=time_of_reporting,
                               has_contact_tracing_app=has_trace_app,
                               testing_delay=self.infection.testing_delay(),
                               additional_attributes=default_additional_attributes,
                               infecting_node=infecting_node)

        # Each house now stores the ID's of which nodes are stored inside the house,
        # so that quarantining can be done at the household level
        household.node_ids.append(new_node)


class NewInfectionIndividualTracingDailyTesting(NewInfection):

    def new_infection(self, time: int, node_count: int, household_id: int,
                      serial_interval=None,
                      infecting_node: Optional[Node] = None,
                      additional_attributes: Optional[dict] = None):
        """Add a new infection to the model and network. Attributes are randomly generated.

        This method passes additional attribute, relevant to the lateral flow testing.

        Args:
            node_count (int): The number of nodes already in the model
            household_id (int): The household id that the node is being added to
            serial_interval ([type]): The serial interval
            infecting_node (Optional[NodeContactModel]): The id of the infecting node
            additional_attributes (Optional[dict]): Additional attributes to be passed
        """

        household = self._network.houses.household(household_id)

        node_will_take_up_lfa_testing = self._infection.will_take_up_lfa_testing()

        if household.being_lateral_flow_tested:

            time_started_lfa_testing = household.being_lateral_flow_tested_start_time

            if node_will_take_up_lfa_testing:
                node_being_lateral_flow_tested = True

            else:
                node_being_lateral_flow_tested = False

        else:
            node_being_lateral_flow_tested = False
            time_started_lfa_testing = float('Inf')

        default_additional_attributes = {
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
                self._infection.will_engage_in_risky_behaviour_while_being_lfa_tested(),
            'propensity_to_miss_lfa_tests': self._infection.propensity_to_miss_lfa_tests()
        }

        if additional_attributes:
            # if new additional attributes are passed, these override the current additional
            # attributes if they are the same value if they are different values, then they
            # are added to the dictionary
            additional_attributes_with_defaults = {**default_additional_attributes,
                                                   **additional_attributes}
        else:
            additional_attributes_with_defaults = default_additional_attributes

        asymptomatic = self._infection.is_asymptomatic_infection()

        # Symptom onset time
        symptom_onset_time = time + self._infection.incubation_period(asymptomatic)

        # If the node is asymptomatic, we need to generate a pseudo symptom onset time
        if asymptomatic:
            pseudo_symptom_onset_time = self._infection.incubation_period(asymptomatic=False)
        else:
            pseudo_symptom_onset_time = symptom_onset_time

        # When a node reports its infection
        if not asymptomatic and np.random.binomial(1, self._infection.infection_reporting_prob) == 1:
            will_report_infection = True
            time_of_reporting = symptom_onset_time + self._infection.reporting_delay(asymptomatic)
        else:
            will_report_infection = False
            time_of_reporting = float('Inf')

        # We assign each node a recovery period of 21 days, after 21 days the probability of
        # causing a new infections is 0, due to the generation time distribution
        recovery_time = time + 14

        household = self._network.houses.household(household_id)

        # If the household has the propensity to use the contact tracing app, decide
        # if the node uses the app.
        if household.propensity_trace_app:
            has_trace_app = self._infection.has_contact_tracing_app()
        else:
            has_trace_app = False

        # in case you want to add non-default additional attributes
        default_additional_attributes = {}

        if additional_attributes_with_defaults:
            default_additional_attributes = {**default_additional_attributes,
                                             **additional_attributes_with_defaults}

        isolation_uptake = self._infection.will_uptake_isolation()

        if household.isolated and isolation_uptake:
            node_is_isolated = True
        else:
            node_is_isolated = False

        new_node = self._network.add_node(
            node_id=node_count,
            time=time,
            household_id=household_id,
            isolated=node_is_isolated,
            will_uptake_isolation=isolation_uptake,
            propensity_imperfect_isolation=self._infection.get_propensity_imperfect_isolation(),
            asymptomatic=asymptomatic,
            contact_traced=household.contact_traced,
            symptom_onset_time=symptom_onset_time,
            pseudo_symptom_onset_time=pseudo_symptom_onset_time,
            serial_interval=serial_interval,
            recovery_time=recovery_time,
            will_report_infection=will_report_infection,
            time_of_reporting=time_of_reporting,
            has_contact_tracing_app=has_trace_app,
            testing_delay=self.infection.testing_delay(),
            additional_attributes=default_additional_attributes,
            infecting_node=infecting_node,
        )

        # Each house now stores the ID's of which nodes are stored inside the house,
        # so that quarantining can be done at the household level
        household.node_ids.append(new_node)
