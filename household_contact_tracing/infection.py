from __future__ import annotations

import numpy as np
import numpy.random as npr
from typing import Type

from household_contact_tracing.behaviours.contact_rate_reduction import ContactRateReduction
from household_contact_tracing.behaviours.new_household import NewHousehold
from household_contact_tracing.behaviours.new_infection import NewInfection
from household_contact_tracing.distributions import current_hazard_rate, current_rate_infection, compute_negbin_cdf
from household_contact_tracing.network import EdgeType, Network, Node
from household_contact_tracing.parameterised import Parameterised


class Infection(Parameterised):
    """
        Logic for creation of infectives and daily increment of infection.

        Attributes
        ----------
        network : Network
            the persistent storage of model data

        Methods
        -------
            initialise(self):
                # Create the starting infectives
            increment(self, time):
                Create a new days worth of infections.

    """

    def __init__(self, network: Network, new_household: Type[NewHousehold],
                 new_infection: Type[NewInfection],
                 contact_rate_reduction: Type[ContactRateReduction], params: dict):
        self.network = network

        # The mean number of contacts made by each household
        self.total_contact_means = [7.238, 10.133, 11.419, 12.844, 14.535, 15.844]

        # Local contact probability:
        self.local_contact_probs = [0, 0.826, 0.795, 0.803, 0.787, 0.819]

        # infection parameters
        self.outside_household_infectivity_scaling = 1.0
        self.overdispersion = 0.32
        self.asymptomatic_relative_infectivity = 0.5
        self.starting_infections = 1
        self.household_pairwise_survival_prob = 0.2

        self.update_params(params)

        household_size = len(self.total_contact_means)

        # Precomputing the cdf's for generating the overdispersed contact data
        self.cdf_dict = {i + 1: compute_negbin_cdf(self.total_contact_means[i], self.overdispersion)
                         for i in range(household_size)}


        self.symptomatic_local_infection_probs = self.compute_hh_infection_probs(self.household_pairwise_survival_prob)
        asymptomatic_household_pairwise_survival_prob = (1 - self.asymptomatic_relative_infectivity
                                                         + self.asymptomatic_relative_infectivity
                                                         * self.household_pairwise_survival_prob)
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

        self.new_household = new_household(self.network, params, self.local_contact_probs,
                                           self.total_contact_means)

        self.new_infection = new_infection(self.network, params)
        self.contact_rate_reduction = contact_rate_reduction(params)

        self.initialise()

    def initialise(self):
        # Create the starting infectives
        for households in range(self.starting_infections):
            new_household = self.new_household.new_household(0, None)
            self.new_infection.new_infection(0, new_household)

    def increment(self, time):
        """Create a new days worth of infections."""
        for node in self.network.active_infections:
            household = node.household

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

            if self.contact_rate_reduction:
                outside_household_contacts = npr.binomial(
                    outside_household_contacts,
                    1 - self.contact_rate_reduction.get_contact_rate_reduction(node)
                )

            # Within household, how many of the infections would cause new infections
            # These contacts may be made with someone who is already infected so they
            # will be thinned again
            local_infection_probs = self.get_infection_prob(local=True,
                                                            infectious_age=days_since_infected,
                                                            asymptomatic=node.asymptomatic)

            local_infective_contacts = npr.binomial(local_contacts, local_infection_probs)

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
                    self.new_within_household_infection(time=time, infecting_node=node)

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
                self.new_outside_household_infection(time=time, infecting_node=node)
                node_time_tuple = (self.network.node_count, time)

                node.spread_to_global_node_time_tuples.append(node_time_tuple)

    def contacts_made_today(self, household_size) -> int:
        """Generates the number of contacts made today by a node, given the house size of the node.
         Uses an overdispersed negative binomial distribution.

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

    def new_outside_household_infection(self, time: int, infecting_node: Node):
        # We assume all new outside household infections are in a new household
        # i.e: You do not infect 2 people in a new household
        # you do not spread the infection to a household that already has an infection
        node_count = self.network.node_count + 1
        infecting_household = infecting_node.household

        # Create a new household, since the infection was outside the household
        new_household = self.new_household.new_household(time=time,
                                                         infected_by=infecting_node.household)

        # We record which house spread to which other house
        infecting_household.spread_to.append(new_household)

        # add a new infection in the house just created
        self.new_infection.new_infection(time, new_household, infecting_node)

        # Add the edge to the graph and give it the default label
        self.network.graph.add_edge(infecting_node.id, node_count)
        self.network.graph.edges[infecting_node.id, node_count].update(
            {"edge_type": EdgeType.default})

    def new_within_household_infection(self, time, infecting_node: Node):
        """Add a new node to the network.

        The new node will be a member of the same household as the infecting node.
        """
        node_count = self.network.node_count + 1

        infecting_node_household = infecting_node.household

        # Adds the new infection to the network
        self.new_infection.new_infection(time, infecting_node_household,
                                         infecting_node=infecting_node)

        # Add the edge to the graph and give it the default label if the house is not
        # traced/isolated
        self.network.graph.add_edge(infecting_node.id, node_count)

        if self.network.node(node_count).household.isolated:
            self.network.graph.edges[infecting_node.id, node_count].update(
                {"edge_type": EdgeType.within_house})
        else:
            self.network.graph.edges[infecting_node.id, node_count].update(
                {"edge_type": EdgeType.default})

        # Decrease the number of susceptibles in that house by 1
        infecting_node_household.susceptibles -= 1

        # We record which edges are within this household for visualisation later on
        infecting_node_household.within_house_edges.append((infecting_node.id, node_count))
