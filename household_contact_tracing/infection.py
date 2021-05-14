from __future__ import annotations
import numpy as np
import numpy.random as npr
from typing import Optional
import sys

from household_contact_tracing.distributions import current_hazard_rate, current_rate_infection, compute_negbin_cdf
from household_contact_tracing.network import Node, EdgeType, Network


class Infection:
    """ 'Context' class for infection processes/strategies (Strategy pattern) """

    def __init__(self, network: Network, new_household: NewHouseholdBehaviour, new_infection: NewInfectionBehaviour,
                 contact_rate_reduction: ContactRateReductionBehaviour, params: dict):
        self._network = network

        # Declare behaviours
        self.new_household_behaviour = new_household
        self.new_infection_behaviour = new_infection
        self.contact_rate_reduction_behaviour = contact_rate_reduction

        # Probability of each household size
        self.house_size_probs = [0.294591195, 0.345336927, 0.154070081, 0.139478886,
                                 0.045067385, 0.021455526]

        # The mean number of contacts made by each household
        self.total_contact_means = [7.238, 10.133, 11.419, 12.844, 14.535, 15.844]

        # Local contact probability:
        self.local_contact_probs = [0, 0.826, 0.795, 0.803, 0.787, 0.819]

        # infection parameters
        self.outside_household_infectivity_scaling = 1.0
        self.overdispersion = 0.32
        self.asymptomatic_prob = 0.5
        self.asymptomatic_relative_infectivity = 0.5
        self.infection_reporting_prob = 1.0
        self.reduce_contacts_by = 0
        self.starting_infections = 1
        self.symptom_reporting_delay = 1
        self.incubation_period_delay = 5
        self.global_contact_reduction_risky_behaviour = 0

        # adherence parameters
        self.node_will_uptake_isolation_prob = 1
        self.propensity_imperfect_quarantine = 0
        self.global_contact_reduction_imperfect_quarantine = 0

        self.node_prob_will_take_up_lfa_testing = 1
        self.propensity_risky_behaviour_lfa_testing = 0
        self.hh_propensity_to_use_trace_app = 1
        self.test_before_propagate_tracing = True
        self.test_delay = 1
        self.prob_has_trace_app = 0
        self.proportion_with_propensity_miss_lfa_tests = 0.

        # Update instance variables with anything in params
        for param_name in self.__dict__:
            if param_name in params:
                self.__dict__[param_name] = params[param_name]

        # Precomputing the cdf's for generating the overdispersed contact data
        self.cdf_dict = {
            1: compute_negbin_cdf(self.total_contact_means[0], self.overdispersion, 100),
            2: compute_negbin_cdf(self.total_contact_means[1], self.overdispersion, 100),
            3: compute_negbin_cdf(self.total_contact_means[2], self.overdispersion, 100),
            4: compute_negbin_cdf(self.total_contact_means[3], self.overdispersion, 100),
            5: compute_negbin_cdf(self.total_contact_means[4], self.overdispersion, 100),
            6: compute_negbin_cdf(self.total_contact_means[5], self.overdispersion, 100)
        }

        # Calculate the expected local contacts
        expected_local_contacts = [self.local_contact_probs[i] * i for i in range(6)]

        # Calculate the expected global contacts
        expected_global_contacts = np.array(self.total_contact_means) - np.array(expected_local_contacts)

        # Size biased distribution of households (choose a node, what is the prob they are in a house size 6, this is
        # biased by the size of the house)
        size_mean_contacts_biased_distribution = [(i + 1) * self.house_size_probs[i] * expected_global_contacts[i] for i
                                                  in range(6)]
        total = sum(size_mean_contacts_biased_distribution)
        self.size_mean_contacts_biased_distribution = [prob / total for prob in size_mean_contacts_biased_distribution]

        self.symptomatic_local_infection_probs = self.compute_hh_infection_probs(
            params["household_pairwise_survival_prob"])
        asymptomatic_household_pairwise_survival_prob = 1 - self.asymptomatic_relative_infectivity \
                                                        + self.asymptomatic_relative_infectivity * \
                                                        params["household_pairwise_survival_prob"]
        self.asymptomatic_local_infection_probs = self.compute_hh_infection_probs(
            asymptomatic_household_pairwise_survival_prob)

        # Precomputing the global infection probabilities
        self.symptomatic_global_infection_probs = []
        self.asymptomatic_global_infection_probs = []
        for day in range(15):
            self.symptomatic_global_infection_probs.append(self.outside_household_infectivity_scaling *
                                                           current_rate_infection(day))
            self.asymptomatic_global_infection_probs.append(self.outside_household_infectivity_scaling *
                                                            self.asymptomatic_relative_infectivity *
                                                            current_rate_infection(day))

        self.initialise()

    @property
    def network(self):
        return self._network

    @property
    def new_household_behaviour(self) -> NewHouseholdBehaviour:
        return self._new_household_behaviour

    @new_household_behaviour.setter
    def new_household_behaviour(self, new_household_behaviour: NewHouseholdBehaviour):
        self._new_household_behaviour = new_household_behaviour
        if self._new_household_behaviour:
            self._new_household_behaviour.infection = self

    @property
    def new_infection_behaviour(self) -> NewInfectionBehaviour:
        return self._new_infection_behaviour


    @new_infection_behaviour.setter
    def new_infection_behaviour(self, new_infection_behaviour: NewInfectionBehaviour):
        self._new_infection_behaviour = new_infection_behaviour
        if self._new_infection_behaviour:
            self._new_infection_behaviour.infection = self

    @property
    def contact_rate_reduction_behaviour(self) -> ContactRateReductionBehaviour:
        return self._contact_rate_reduction_behaviour

    @contact_rate_reduction_behaviour.setter
    def contact_rate_reduction_behaviour(self, contact_rate_reduction_behaviour: ContactRateReductionBehaviour):
        self._contact_rate_reduction_behaviour = contact_rate_reduction_behaviour
        if self._contact_rate_reduction_behaviour:
            self._contact_rate_reduction_behaviour.infection = self

    def new_household(self,
                      time,
                      new_household_number,
                      generation,
                      infected_by,
                      infected_by_node,
                      additional_attributes=None):
        if self.new_household_behaviour:
            self.new_household_behaviour.new_household(time,
                                                       new_household_number,
                                                       generation,
                                                       infected_by,
                                                       infected_by_node,
                                                       additional_attributes)

    def new_infection(self,
                      time: int,
                      node_count: int,
                      generation: int,
                      household_id: int,
                      test_delay: int = 0,
                      serial_interval=None,
                      infecting_node: Optional[Node] = None,
                      additional_attributes=None):
        if self.new_infection_behaviour:
            self.new_infection_behaviour.new_infection(time,
                                                       node_count,
                                                       generation,
                                                       household_id,
                                                       serial_interval,
                                                       infecting_node,
                                                       additional_attributes)

    def get_contact_rate_reduction(self, node: Node) -> int:
        if self.contact_rate_reduction_behaviour:
            return self.contact_rate_reduction_behaviour.get_contact_rate_reduction(node)

    def initialise(self):

        # Create first household
        # Initial values
        house_id = 0
        generation = 0

        # Create the starting infectives
        for time in range(self.starting_infections):
            house_id += 1
            node_id = self.network.node_count + 1
            self.new_household(time, house_id, 1, None, None, None)
            self.new_infection(time, node_id, generation, house_id)

    def increment(self, time):
        """
        Creates a new days worth of infections
        """

        for node in self.network.active_infections:
            household = node.household()

            # Extracting useful parameters from the node
            days_since_infected = time - node.time_infected

            outside_household_contacts = -1
            local_contacts = -1

            while outside_household_contacts < 0:
                # The number of contacts made that day
                contacts_made = self.contacts_made_today(household.size)

                # How many of the contacts are within the household
                local_contacts = npr.binomial(household.size - 1,
                                              self.local_contact_probs[household.size - 1])

                # How many of the contacts are outside household contacts
                outside_household_contacts = contacts_made - local_contacts

            if self.contact_rate_reduction_behaviour:
                outside_household_contacts = npr.binomial(
                    outside_household_contacts,
                    1 - self.get_contact_rate_reduction(node)
                )

            # Within household, how many of the infections would cause new infections
            # These contacts may be made with someone who is already infected so they
            # will be thinned again
            local_infection_probs = self.get_infection_prob(local=True,
                                                            infectious_age=days_since_infected,
                                                            asymptomatic=node.asymptomatic)

            local_infective_contacts = npr.binomial(
                local_contacts,
                local_infection_probs
            )

            for _ in range(local_infective_contacts):
                # A further thinning has to happen since each attempt may choose an
                # already infected person. That is to say, if everyone in your house is infected,
                # you have 0 chance to infect a new person in your house

                # A one represents a susceptibles node in the household
                # A 0 represents an infected member of the household
                # We choose a random subset of this vector of length local_infective_contacts
                # to determine infections, i.e we are choosing without replacement
                household_composition = [1] * household.susceptibles + [0] * (
                            household.size - 1 - household.susceptibles)
                within_household_new_infections = sum(
                    npr.choice(household_composition, local_infective_contacts, replace=False))

                # If the within household infection is successful:
                for _ in range(within_household_new_infections):
                    self.new_within_household_infection(
                        time=time,
                        infecting_node=node,
                        serial_interval=days_since_infected
                    )

            # Update how many contacts the node made
            node.outside_house_contacts_made += outside_household_contacts

            # How many outside household contacts cause new infections
            global_infection_probs = self.get_infection_prob(local=False,
                                                             infectious_age=days_since_infected,
                                                             asymptomatic=node.asymptomatic)
            outside_household_new_infections = npr.binomial(
                outside_household_contacts,
                global_infection_probs
            )

            for _ in range(outside_household_new_infections):
                self.new_outside_household_infection(
                    time=time,
                    infecting_node=node,
                    serial_interval=days_since_infected)

                node_time_tuple = (self.network.node_count, time)

                node.spread_to_global_node_time_tuples.append(node_time_tuple)

    def is_asymptomatic_infection(self) -> bool:
        return npr.binomial(1, self.asymptomatic_prob) == 1

    def incubation_period(self, asymptomatic: bool) -> int:
        if asymptomatic:
            return sys.maxsize
        else:
            return round(self.incubation_period_delay)

    def size_of_household(self) -> int:
        """Generates a random household size

        Returns:
        household_size {int}
        """
        return npr.choice([1, 2, 3, 4, 5, 6], p=self.size_mean_contacts_biased_distribution)


    def contacts_made_today(self, household_size) -> int:
        """Generates the number of contacts made today by a node, given the house size of the node. Uses an
        overdispersed negative binomial distribution.

        Arguments:
            house_size {int} -- size of the nodes household
        """
        random = npr.uniform()
        cdf = self.cdf_dict[household_size]
        obs = sum([int(cdf[i] < random) for i in range(100)])
        return obs

    def compute_hh_infection_probs(self, pairwise_survival_prob: float) -> np.ndarray:
        # Precomputing the infection probabilities for the within household epidemics.
        contact_prob = 0.8
        day_0_infection_prob = current_hazard_rate(0, pairwise_survival_prob) / contact_prob
        infection_probs = np.array(day_0_infection_prob)
        for day in range(1, 15):
            survival_function = (1 - infection_probs*contact_prob).prod()
            hazard = current_hazard_rate(day, pairwise_survival_prob)
            current_prob_infection = hazard * (survival_function / contact_prob)
            infection_probs = np.append(infection_probs, current_prob_infection)
        return infection_probs

    def reporting_delay(self, asymptomatic: bool) -> int:
        if asymptomatic:
            return sys.maxsize
        else:
            return round(self.symptom_reporting_delay)

    def will_uptake_isolation(self) -> bool:
        """Based on the node_will_uptake_isolation_prob, return a bool
        where True implies they do take up isolation and False implies they do not uptake isolation

        Returns:
            bool: If True they uptake isolation, if False they do not uptake isolation
        """
        return npr.choice([True, False], p=(self.node_will_uptake_isolation_prob, 1 -
                                            self.node_will_uptake_isolation_prob))

    def get_propensity_imperfect_isolation(self) -> bool:
        return npr.choice([True, False], p=(self.propensity_imperfect_quarantine, 1 -
                                            self.propensity_imperfect_quarantine))

    def get_infection_prob(self, local: bool, infectious_age: int, asymptomatic: bool) -> float:
        """Get the current probability per global infectious contact

        Args:
            local (bool): Is the contact a local contact or a global contact?
            infectious_age (int): The current infectious age of the potential infector
            asymptomatic (bool): Whether or not the node is asymptomatic

        Returns:
            float: The probability the contact spreads the infection
        """

        if local:
            if asymptomatic:
                return self.asymptomatic_local_infection_probs[infectious_age]
            else:
                return self.symptomatic_local_infection_probs[infectious_age]
        else:
            if asymptomatic:
                return self.asymptomatic_global_infection_probs[infectious_age]
            else:
                return self.symptomatic_global_infection_probs[infectious_age]

    def new_outside_household_infection(self, time: int, infecting_node: Node,
                                        serial_interval: Optional[int]):
        # We assume all new outside household infections are in a new household
        # i.e: You do not infect 2 people in a new household
        # you do not spread the infection to a household that already has an infection
        house_id = self.network.house_count + 1
        node_count = self.network.node_count + 1
        infecting_household = infecting_node.household()

        # We record which node caused this infection
        infecting_node.spread_to.append(node_count)

        # We record which house spread to which other house
        infecting_household.spread_to_ids.append(house_id)

        # Create a new household, since the infection was outside the household
        self.new_household(time=time,
                           new_household_number=house_id,
                           generation=infecting_household.generation + 1,
                           infected_by=infecting_node.household_id,
                           infected_by_node=infecting_node.node_id)

        # add a new infection in the house just created
        self.new_infection(time=time,
                           node_count=node_count,
                           generation=infecting_node.generation + 1,
                           household_id=house_id,
                           serial_interval=serial_interval,
                           infecting_node=infecting_node)

        # Add the edge to the graph and give it the default label
        self._network.graph.add_edge(infecting_node.node_id, node_count)
        self._network.graph.edges[infecting_node.node_id, node_count].update(
            {"edge_type": EdgeType.default.name})

    def new_within_household_infection(self, time, infecting_node: Node,
                                       serial_interval: Optional[int]):
        """Add a new node to the network.

        The new node will be a member of the same household as the infecting node.
        """
        node_count = self._network.node_count + 1

        # We record which node caused this infection
        infecting_node.spread_to.append(node_count)

        infecting_node_household = infecting_node.household()

        # Adds the new infection to the network
        self.new_infection(time=time,
                           node_count=node_count,
                           generation=infecting_node.generation + 1,
                           household_id=infecting_node_household.house_id,
                           serial_interval=serial_interval,
                           infecting_node=infecting_node)

        # Add the edge to the graph and give it the default label if the house is not
        # traced/isolated
        self.network.graph.add_edge(infecting_node.node_id, node_count)

        if self.network.node(node_count).household().isolated:
            self.network.graph.edges[infecting_node.node_id, node_count].update(
                {"edge_type": EdgeType.within_house.name})
        else:
            self.network.graph.edges[infecting_node.node_id, node_count].update(
                {"edge_type": EdgeType.default.name})

        # Decrease the number of susceptibles in that house by 1
        infecting_node_household.susceptibles -= 1

        # We record which edges are within this household for visualisation later on
        infecting_node_household.within_house_edges.append((infecting_node.node_id, node_count))

    def will_take_up_lfa_testing(self) -> bool:
        return npr.binomial(1, self.node_prob_will_take_up_lfa_testing) == 1

    def will_engage_in_risky_behaviour_while_being_lfa_tested(self):
        """Will the node engage in more risky behaviour if they are being LFA tested?
        """
        if npr.binomial(1, self.propensity_risky_behaviour_lfa_testing) == 1:
            return True
        else:
            return False

    def propensity_to_miss_lfa_tests(self) -> bool:
        return npr.binomial(1, self.proportion_with_propensity_miss_lfa_tests) == 1

    def has_contact_tracing_app(self) -> bool:
        return npr.binomial(1, self.prob_has_trace_app) == 1

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


class NewHouseholdBehaviour:
    def __init__(self, network: Network):
        self._network = network
        self._infection = None

    @property
    def infection(self) -> Infection:
        return self._infection

    @infection.setter
    def infection(self, infection: Infection):
        self._infection = infection

    def new_household(self,
                      time: int,
                      new_household_number: int,
                      generation: int,
                      infected_by: int,
                      infected_by_node: int,
                      additional_attributes: Optional[dict] = None):
        pass


class NewHousehold(NewHouseholdBehaviour):

    def new_household(self,
                      time: int,
                      new_household_number: int,
                      generation: int,
                      infected_by: int,
                      infected_by_node: int,
                      additional_attributes: Optional[dict] = None):
        """Adds a new household to the household dictionary

        Arguments:
            new_household_number {int} -- The house id
            generation {int} -- The household generation of this household
            infected_by {int} -- Which household spread the infection to this household
            infected_by_node {int} -- Which node spread the infection to this household
        """
        house_size = self._infection.size_of_household()

        self._network.houses.add_household(
            house_id=new_household_number,
            house_size=house_size,
            time_infected=time,
            generation=generation,
            infected_by=infected_by,
            infected_by_node=infected_by_node,
            propensity_trace_app=self.infection.hh_propensity_use_trace_app(),
            additional_attributes=additional_attributes
        )


class NewHouseholdContactModelTest(NewHousehold):

    def new_household(self,
                      time: int,
                      new_household_number: int,
                      generation: int,
                      infected_by: int,
                      infected_by_node: int,
                      additional_attributes: Optional[dict] = None):

        super().new_household(
            time,
            new_household_number=new_household_number,
            generation=generation,
            infected_by=infected_by,
            infected_by_node=infected_by_node,
            additional_attributes={
                'being_lateral_flow_tested': False,
                'being_lateral_flow_tested_start_time': None,
                'applied_policy_for_household_contacts_of_a_positive_case': False
            }
        )


class NewInfectionBehaviour:
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
                      generation: int,
                      household_id: int,
                      serial_interval=None,
                      infecting_node: Optional[Node] = None,
                      additional_attributes: Optional[dict] = None):
        pass


class NewInfectionHousehold(NewInfectionBehaviour):

    def new_infection(self,
                      time: int,
                      node_count: int,
                      generation: int,
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
        if not asymptomatic and npr.binomial(1, self._infection.infection_reporting_prob) == 1:
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

        self._network.add_node(node_id=node_count, time=time, generation=generation,
                               household=household_id, isolated=node_is_isolated,
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
        household.node_ids.append(node_count)


class NewInfectionContactModelTest(NewInfectionBehaviour):

    def new_infection(self, time: int, node_count: int, generation: int, household_id: int,
                      serial_interval=None,
                      infecting_node: Optional[Node] = None,
                      additional_attributes: Optional[dict] = None):
        """Add a new infection to the model and network. Attributes are randomly generated.

        This method passes additional attribute, relevant to the lateral flow testing.

        Args:
            node_count (int): The number of nodes already in the model
            generation (int): The generation of the node
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
        if not asymptomatic and npr.binomial(1, self._infection.infection_reporting_prob) == 1:
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

        self._network.add_node(
            node_id=node_count,
            time=time,
            generation=generation,
            household=household_id,
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
        household.node_ids.append(node_count)


class ContactRateReductionBehaviour:
    def __init__(self):
        self._infection = None

    @property
    def infection(self) -> Infection:
        return self._infection

    @infection.setter
    def infection(self, infection: Infection):
        self._infection = infection

    def get_contact_rate_reduction(self, node) -> int:
        pass


class ContactRateReductionHousehold(ContactRateReductionBehaviour):

    def get_contact_rate_reduction(self, node) -> int:
        """Returns a contact rate reduction, depending upon a nodes current status and various
        isolation parameters
        """

        if node.isolated and node.propensity_imperfect_isolation:
            return self._infection.global_contact_reduction_imperfect_quarantine
        elif node.isolated and not node.propensity_imperfect_isolation:
            # return 1 means 100% of contacts are stopped
            return 1
        else:
            return self._infection.reduce_contacts_by


class ContactRateReductionContactModelTest(ContactRateReductionBehaviour):

    def get_contact_rate_reduction(self, node) -> int:
        """This method overrides the default behaviour. Previously the override behaviour allowed
        he global contact reduction to vary by household size.

        We override this behaviour, so that we can vary the global contact reduction by whether a
        node is isolating or being lfa tested or whether they engage in risky behaviour while they
         are being lfa tested.

        Remember that a contact rate reduction of 1 implies that 100% of contacts are stopped.
        """
        # the isolated status should never apply to an individual who will not uptake isolation

        if node.isolated and not node.propensity_imperfect_isolation:
            # perfect isolation
            return 1

        elif node.isolated and node.propensity_imperfect_isolation:
            # imperfect isolation
            return self._infection.global_contact_reduction_imperfect_quarantine

        elif node.being_lateral_flow_tested and node.propensity_risky_behaviour_lfa_testing:
            # engaging in risky behaviour while testing negative
            return self._infection.global_contact_reduction_risky_behaviour

        else:
            # normal levels of social distancing
            return self._infection.reduce_contacts_by
