import numpy as np
import numpy.random as npr
from typing import Optional
import sys

from household_contact_tracing.distributions import current_hazard_rate, current_rate_infection, compute_negbin_cdf
from household_contact_tracing.network import Node, EdgeType, Network


class Infection:
    """ Class for Infection processes """

    def __init__(self, network: Network, params: dict):
        self._network = network
        self._new_household = None
        self._new_infection = None
        self._contact_rate_reduction = None

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

        # adherence parameters
        self.node_will_uptake_isolation_prob = 1
        self.propensity_imperfect_quarantine = 0
        self.global_contact_reduction_imperfect_quarantine = 0

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
        asymptomatic_household_pairwise_survival_prob = 1 - self.asymptomatic_relative_infectivity + self.asymptomatic_relative_infectivity * \
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

    @property
    def network(self):
        return self._network

    @property
    def new_household(self):
        return self._new_household

    @new_household.setter
    def new_household(self, fn):
        self._new_household = fn

    @property
    def new_infection(self):
        return self._new_infection

    @new_infection.setter
    def new_infection(self, fn):
        self._new_infection = fn

    @property
    def contact_rate_reduction(self):
        return self._contact_rate_reduction

    @contact_rate_reduction.setter
    def contact_rate_reduction(self, fn):
        self._contact_rate_reduction = fn

    def reset(self):

        # Create first household
        # Initial values
        house_id = 0
        generation = 0

        # Create the starting infectives
        for _ in range(self.starting_infections):
            house_id += 1
            node_id = self.network.node_count + 1
            if self.new_household:
                self.new_household(house_id, 1, None, None)
            if self.new_infection:
                self.new_infection(node_id, generation, house_id)

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
                local_contacts = npr.binomial(household.size - 1, self.local_contact_probs[household.size - 1])

                # How many of the contacts are outside household contacts
                outside_household_contacts = contacts_made - local_contacts

            if self.contact_rate_reduction:
                outside_household_contacts = npr.binomial(
                    outside_household_contacts,
                    1 - self.contact_rate_reduction(node)
                )

            # Within household, how many of the infections would cause new infections
            # These contacts may be made with someone who is already infected, and so they will again be thinned
            local_infection_probs = self.get_infection_prob(local=True, infectious_age=days_since_infected,
                                                            asymptomatic=node.asymptomatic)

            local_infective_contacts = npr.binomial(
                local_contacts,
                local_infection_probs
            )

            for _ in range(local_infective_contacts):
                # A further thinning has to happen since each attempt may choose an already infected person
                # That is to say, if everyone in your house is infected, you have 0 chance to infect a new person in your house

                # A one represents a susceptibles node in the household
                # A 0 represents an infected member of the household
                # We choose a random subset of this vector of length local_infective_contacts to determine infections
                # i.e we are choosing without replacement
                household_composition = [1] * household.susceptibles + [0] * (
                            household.size - 1 - household.susceptibles)
                within_household_new_infections = sum(
                    npr.choice(household_composition, local_infective_contacts, replace=False))

                # If the within household infection is successful:
                for _ in range(within_household_new_infections):
                    self.new_within_household_infection(
                        infecting_node=node,
                        serial_interval=days_since_infected
                    )

            # Update how many contacts the node made
            node.outside_house_contacts_made += outside_household_contacts

            # How many outside household contacts cause new infections
            global_infection_probs = self.get_infection_prob(local=False, infectious_age=days_since_infected,
                                                             asymptomatic=node.asymptomatic)
            outside_household_new_infections = npr.binomial(
                outside_household_contacts,
                global_infection_probs
            )

            for _ in range(outside_household_new_infections):
                self.new_outside_household_infection(
                    infecting_node=node,
                    serial_interval=days_since_infected)

                node_time_tuple = (self.network.node_count, time)

                node.spread_to_global_node_time_tuples.append(node_time_tuple)

    def is_asymptomatic_infection(self) -> bool:
        return npr.binomial(1, self.asymptomatic_prob) == 1

    def incubation_period(self, asymptomatic: bool) -> int:
        if asymptomatic:
            return float('Inf')
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

    def compute_hh_infection_probs(self, pairwise_survival_prob: float) -> list:
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

    def reporting_delay(self, asymptomatic: bool):
        if asymptomatic:
            return float('Inf')
        else:
            return round(self.symptom_reporting_delay)

    def will_uptake_isolation(self) -> bool:
        """Based on the node_will_uptake_isolation_prob, return a bool
        where True implies they do take up isolation and False implies they do not uptake isolation

        Returns:
            bool: If True they uptake isolation, if False they do not uptake isolation
        """
        return npr.choice([True, False], p = (self.node_will_uptake_isolation_prob, 1 -
                                              self.node_will_uptake_isolation_prob))

    def get_propensity_imperfect_isolation(self) -> bool:
        return npr.choice([True, False], p = (self.propensity_imperfect_quarantine, 1 -
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

    def new_outside_household_infection(self, infecting_node: 'Node', serial_interval: Optional[int]):
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
        self.new_household(new_household_number=house_id,
                           generation=infecting_household.generation + 1,
                           infected_by=infecting_node.household_id,
                           infected_by_node=infecting_node.node_id)

        # add a new infection in the house just created
        self.new_infection(node_count=node_count,
                           generation=infecting_node.generation + 1,
                           household_id=house_id,
                           serial_interval=serial_interval,
                           infecting_node=infecting_node)

        # Add the edge to the graph and give it the default label
        self.network.graph.add_edge(infecting_node.node_id, node_count)
        self.network.graph.edges[infecting_node.node_id, node_count].update(
            {"edge_type": EdgeType.default.name})

    def new_within_household_infection(self, infecting_node: 'Node', serial_interval: Optional[int]):
        # Add a new node to the network, it will be a member of the same household that the node that infected it was
        node_count = self.network.node_count + 1

        # We record which node caused this infection
        infecting_node.spread_to.append(node_count)

        infecting_node_household = infecting_node.household()

        # Adds the new infection to the network
        self.new_infection(node_count=node_count,
                           generation=infecting_node.generation + 1,
                           household_id=infecting_node_household.house_id,
                           serial_interval=serial_interval,
                           infecting_node=infecting_node)

        # Add the edge to the graph and give it the default label if the house is not traced/isolated
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
