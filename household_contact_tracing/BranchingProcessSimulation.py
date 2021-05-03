from typing import List, Optional, Callable
import numpy as np
import numpy.random as npr

from household_contact_tracing.distributions import current_hazard_rate, current_rate_infection, compute_negbin_cdf
from household_contact_tracing.network import Network, Household, Node, EdgeType, graphs_isomorphic
from household_contact_tracing.bp_simulation_model import BPSimulationModel
from household_contact_tracing.parameters import validate_parameters
from household_contact_tracing.simulation_states import RunningState
from household_contact_tracing.infection import *
from household_contact_tracing.contact_tracing import *


class household_sim_contact_tracing(BPSimulationModel):
    # Local contact probability:
    local_contact_probs = [0, 0.826, 0.795, 0.803, 0.787, 0.819]

    # The mean number of contacts made by each household
    total_contact_means = [7.238, 10.133, 11.419, 12.844, 14.535, 15.844]

    def __init__(self, params: dict):

        """Initializes a household branching process epidemic. Various contact tracing strategies can be utilized
        in an attempt to control the epidemic.

        Args:
            params (dict): A dictionary of parameters that are used in the model.
        """

        # Call parent init
        BPSimulationModel.__init__(self)

        # Parse parameters against schema to check they are valid
        validate_parameters(params, "./schemas/household_sim_contact_tracing.json")

        self._network = Network()
        self._infection = Infection(self._network)
        self._contact_tracing = ContactTracing(self._network)

        # Probability of each household size
        if "house_size_probs" in params:
            self.house_size_probs = params["house_size_probs"]
        else:
            self.house_size_probs = [0.294591195, 0.345336927, 0.154070081, 0.139478886, 0.045067385, 0.021455526]

        # Calculate the expected local contacts
        expected_local_contacts = [self.local_contact_probs[i] * i for i in range(6)]

        # Calculate the expected global contacts
        expected_global_contacts = np.array(self.total_contact_means) - np.array(expected_local_contacts)

        # Size biased distribution of households (choose a node, what is the prob they are in a house size 6, this is
        # biased by the size of the house)
        size_mean_contacts_biased_distribution = [(i + 1) * self.house_size_probs[i] * expected_global_contacts[i] for i in range(6)]
        total = sum(size_mean_contacts_biased_distribution)
        self.size_mean_contacts_biased_distribution = [prob / total for prob in size_mean_contacts_biased_distribution]

        # Parameter Inputs:

        # infection parameters
        self.outside_household_infectivity_scaling = params["outside_household_infectivity_scaling"]
        self.overdispersion = params["overdispersion"]
        self.asymptomatic_prob = params["asymptomatic_prob"]
        self.asymptomatic_relative_infectivity = params["asymptomatic_relative_infectivity"]
        self.infection_reporting_prob = params["infection_reporting_prob"]
        if "reduce_contacts_by" in params:
            self.reduce_contacts_by = params["reduce_contacts_by"]
        else:
            self.reduce_contacts_by = 0
        if "starting_infections" in params:
            self.starting_infections = params["starting_infections"]
        else:
            self.starting_infections = 1
        self.symptom_reporting_delay = params["symptom_reporting_delay"]
        self.incubation_period_delay = params["incubation_period_delay"]

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

        # isolation or quarantine parameters
        if "quarantine_duration" in params:
            self.quarantine_duration = params["quarantine_duration"]
        else:
            self.quarantine_duration = 14
        if "self_isolation_duration" in params:
            self.self_isolation_duration = params["self_isolation_duration"]
        else:
            self.self_isolation_duration = 7

        # adherence parameters
        if "node_will_uptake_isolation_prob" in params:
            self.node_will_uptake_isolation_prob = params["node_will_uptake_isolation_prob"]
        else:
            self.node_will_uptake_isolation_prob = 1
        if "propensity_imperfect_quarantine" in params:
            self.propensity_imperfect_quarantine = params["propensity_imperfect_quarantine"]
        else:
            self.propensity_imperfect_quarantine = 0
        if "global_contact_reduction_imperfect_quarantine" in params:
            self.global_contact_reduction_imperfect_quarantine = params["global_contact_reduction_imperfect_quarantine"]
        else:
            self.global_contact_reduction_imperfect_quarantine = 0

        self.symptomatic_local_infection_probs = self.compute_hh_infection_probs(params["household_pairwise_survival_prob"])
        asymptomatic_household_pairwise_survival_prob = 1 - self.asymptomatic_relative_infectivity + self.asymptomatic_relative_infectivity * params["household_pairwise_survival_prob"]
        self.asymptomatic_local_infection_probs = self.compute_hh_infection_probs(asymptomatic_household_pairwise_survival_prob)

        # Precomputing the cdf's for generating the overdispersed contact data
        self.cdf_dict = {
            1: compute_negbin_cdf(self.total_contact_means[0], self.overdispersion, 100),
            2: compute_negbin_cdf(self.total_contact_means[1], self.overdispersion, 100),
            3: compute_negbin_cdf(self.total_contact_means[2], self.overdispersion, 100),
            4: compute_negbin_cdf(self.total_contact_means[3], self.overdispersion, 100),
            5: compute_negbin_cdf(self.total_contact_means[4], self.overdispersion, 100),
            6: compute_negbin_cdf(self.total_contact_means[5], self.overdispersion, 100)
        }

        # Precomputing the global infection probabilities
        self.symptomatic_global_infection_probs = []
        self.asymptomatic_global_infection_probs = []
        for day in range(15):
            self.symptomatic_global_infection_probs.append(self.outside_household_infectivity_scaling *
                                                           current_rate_infection(day))
            self.asymptomatic_global_infection_probs.append(self.outside_household_infectivity_scaling *
                                                            self.asymptomatic_relative_infectivity *
                                                            current_rate_infection(day))

        # Calls the simulation reset function, which creates all the required dictionaries
        self.initialise_simulation()

    @property
    def network(self):
        return self._network

    @property
    def infection(self) -> InfectionBehaviourInterface:
        return self._infection

    @infection.setter
    def infection(self, infection: InfectionBehaviourInterface):
        self._infection = infection

    @property
    def contact_tracing(self) -> ContactTracingBehaviourInterface:
        return self._contact_tracing

    @contact_tracing.setter
    def contact_tracing(self, contact_tracing: ContactTracingBehaviourInterface):
        self._contact_tracing = contact_tracing

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

    def contact_trace_delay(self, app_traced_edge) -> int:
        if app_traced_edge:
            return 0
        else:
            return round(self.contact_trace_delay)

    def incubation_period(self, asymptomatic: bool) -> int:
        if asymptomatic:
            return float('Inf')
        else:
            return round(self.incubation_period_delay)

    def testing_delay(self) -> int:
        if self.test_before_propagate_tracing is False:
            return 0
        else:
            return round(self.test_delay)

    def is_asymptomatic_infection(self) -> bool:
        return npr.binomial(1, self.asymptomatic_prob) == 1

    def reporting_delay(self, asymptomatic: bool):
        if asymptomatic:
            return float('Inf')
        else:
            return round(self.symptom_reporting_delay)

    def hh_propensity_use_trace_app(self) -> bool:
        if npr.binomial(1, self.hh_propensity_to_use_trace_app) == 1:
            return True
        else:
            return False

    def will_uptake_isolation(self) -> bool:
        """Based on the node_will_uptake_isolation_prob, return a bool
        where True implies they do take up isolation and False implies they do not uptake isolation

        Returns:
            bool: If True they uptake isolation, if False they do not uptake isolation
        """
        return npr.choice([True, False], p = (self.node_will_uptake_isolation_prob, 1 - self.node_will_uptake_isolation_prob))

    def get_propensity_imperfect_isolation(self) -> bool:
        return npr.choice([True, False], p = (self.propensity_imperfect_quarantine, 1 - self.propensity_imperfect_quarantine))
    
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

    def size_of_household(self) -> int:
        """Generates a random household size

        Returns:
        household_size {int}
        """
        return npr.choice([1, 2, 3, 4, 5, 6], p=self.size_mean_contacts_biased_distribution)

    def has_contact_tracing_app(self) -> bool:
        return npr.binomial(1, self.prob_has_trace_app) == 1



    def new_infection(
        self,
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
        asymptomatic = self.is_asymptomatic_infection()

        # Symptom onset time
        symptom_onset_time = self.time + self.incubation_period(asymptomatic)

        # If the node is asymptomatic, we need to generate a pseudo symptom onset time
        if asymptomatic:
            pseudo_symptom_onset_time = self.incubation_period(asymptomatic=False)
        else:
            pseudo_symptom_onset_time = symptom_onset_time

        # When a node reports its infection
        if not asymptomatic and npr.binomial(1, self.infection_reporting_prob) == 1:
            will_report_infection = True
            time_of_reporting = symptom_onset_time + self.reporting_delay(asymptomatic)
        else:
            will_report_infection = False
            time_of_reporting = float('Inf')

        # We assign each node a recovery period of 21 days, after 21 days the probability of
        # causing a new infections is 0, due to the generation time distribution
        recovery_time = self.time + 14

        household = self.network.houses.household(household_id)

        # If the household has the propensity to use the contact tracing app, decide
        # if the node uses the app.
        if household.propensity_trace_app:
            has_trace_app = self.has_contact_tracing_app()
        else:
            has_trace_app = False

        # in case you want to add non-default additional attributes
        default_additional_attributes = {}

        if additional_attributes:
            default_additional_attributes = {**default_additional_attributes, **additional_attributes}

        isolation_uptake = self.will_uptake_isolation()

        if household.isolated and isolation_uptake:
            node_is_isolated = True
        else:
            node_is_isolated = False

        self.network.add_node(
            node_id=node_count,
            time=self.time,
            generation=generation,
            household=household_id,
            isolated=node_is_isolated,
            will_uptake_isolation=isolation_uptake,
            propensity_imperfect_isolation=self.get_propensity_imperfect_isolation(),
            asymptomatic=asymptomatic,
            contact_traced=household.contact_traced,
            symptom_onset_time=symptom_onset_time,
            pseudo_symptom_onset_time=pseudo_symptom_onset_time,
            serial_interval=serial_interval,
            recovery_time=recovery_time,
            will_report_infection=will_report_infection,
            time_of_reporting=time_of_reporting,
            has_contact_tracing_app=has_trace_app,
            testing_delay=self.testing_delay(),
            additional_attributes=default_additional_attributes,
            infecting_node=infecting_node,
        )

        # Each house now stores the ID's of which nodes are stored inside the house,
        # so that quarantining can be done at the household level
        household.node_ids.append(node_count)

    def new_household(
        self,
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
        house_size = self.size_of_household()

        propensity_trace_app = self.hh_propensity_use_trace_app()

        self.network.houses.add_household(
            house_id=new_household_number,
            house_size=house_size,
            time_infected=self.time,
            generation=generation,
            infected_by=infected_by,
            infected_by_node=infected_by_node,
            propensity_trace_app=propensity_trace_app,
            additional_attributes=additional_attributes
        )


    def get_contact_rate_reduction(self, node: Node):
        """Returns a contact rate reduction, depending upon a nodes current status and various isolation
        parameters
        """

        if node.isolated and node.propensity_imperfect_isolation:
            return self.global_contact_reduction_imperfect_quarantine
        elif node.isolated and not node.propensity_imperfect_isolation:
            # return 1 means 100% of contacts are stopped
            return 1
        else:
            return self.reduce_contacts_by
            

    def get_hh_infection_prob(self, infectious_age: int, asymptomatic: bool) -> float:
        """Returns the current probability per local infectious contact.

        Args:
            infectious_age (int): The current infectious age
            asymptomatic (bool): Whether or not the node is asymptomatic

        Returns:
            float: The probability the contact spreads the infection
        """

        if asymptomatic:
            self.asymptomatic_hh_infection_probs[infectious_age]
        else:
            self.symptomatic_hh_infection_probs[infectious_age]

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

    def increment_infection(self):
        """
        Creates a new days worth of infections
        """

        for node in self.network.active_infections:
            household = node.household()

            # Extracting useful parameters from the node
            days_since_infected = self.time - node.time_infected

            outside_household_contacts = -1
            local_contacts = -1

            while outside_household_contacts < 0:

                # The number of contacts made that day
                contacts_made = self.contacts_made_today(household.size)

                # How many of the contacts are within the household
                local_contacts = npr.binomial(household.size - 1, self.local_contact_probs[household.size - 1])

                # How many of the contacts are outside household contacts
                outside_household_contacts = contacts_made - local_contacts

            outside_household_contacts = npr.binomial(
                outside_household_contacts,
                1 - self.get_contact_rate_reduction(node = node)
            )

            # Within household, how many of the infections would cause new infections
            # These contacts may be made with someone who is already infected, and so they will again be thinned
            local_infection_probs = self.get_infection_prob(local = True, infectious_age = days_since_infected, asymptomatic =  node.asymptomatic)

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
                household_composition = [1]*household.susceptibles + [0]*(household.size - 1 - household.susceptibles)
                within_household_new_infections = sum(npr.choice(household_composition, local_infective_contacts, replace=False))

                # If the within household infection is successful:
                for _ in range(within_household_new_infections):
                    self.new_within_household_infection(
                        infecting_node=node,
                        serial_interval=days_since_infected
                    )

            # Update how many contacts the node made
            node.outside_house_contacts_made += outside_household_contacts

            # How many outside household contacts cause new infections
            global_infection_probs = self.get_infection_prob(local = False, infectious_age = days_since_infected, asymptomatic =  node.asymptomatic)
            outside_household_new_infections = npr.binomial(
                outside_household_contacts,
                global_infection_probs
            )

            for _ in range(outside_household_new_infections):
                self.new_outside_household_infection(
                    infecting_node=node,
                    serial_interval=days_since_infected)

                node_time_tuple = (self.network.node_count, self.time)

                node.spread_to_global_node_time_tuples.append(node_time_tuple)

    def new_within_household_infection(self, infecting_node: Node, serial_interval: Optional[int]):
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
            self.network.graph.edges[infecting_node.node_id, node_count].update({"edge_type": EdgeType.within_house.name})
        else:
            self.network.graph.edges[infecting_node.node_id, node_count].update({"edge_type": EdgeType.default.name})

        # Decrease the number of susceptibles in that house by 1
        infecting_node_household.susceptibles -= 1

        # We record which edges are within this household for visualisation later on
        infecting_node_household.within_house_edges.append((infecting_node.node_id, node_count))

    def new_outside_household_infection(self, infecting_node: Node, serial_interval: Optional[int]):
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
        self.network.graph.edges[infecting_node.node_id, node_count].update({"edge_type": EdgeType.default.name})

    def update_isolation(self):
        # Update the contact traced status for all households that have had the contact tracing process get there
        [
            self.contact_trace_household(household)
            for household in self.network.houses.all_households()
            if household.time_until_contact_traced <= self.time
            and not household.contact_traced
        ]

        # Isolate all non isolated households where the infection has been reported (excludes those who will not take up isolation if prob <1)
        [
            self.isolate_household(node.household())
            for node in self.network.all_nodes()
            if node.time_of_reporting + node.testing_delay == self.time
            and not node.household().isolated
            and not node.household().contact_traced
        ]

    def increment_contact_tracing(self):
        """
        Performs a days worth of contact tracing by:
        * Looks for houses where the contact tracing delay is over and moves them to the contact traced state
        * Looks for houses in the contact traced state, and checks them for symptoms. If any of them have symptoms, the house is isolated

        The isolation function also assigns contact tracing times to any houses that had contact with that household

        For each node that is contact traced
        """

        # Isolate all households under observation that now display symptoms (excludes those who will not take up isolation if prob <1)
        [
            self.isolate_household(node.household())
            for node in self.network.all_nodes()
            if node.symptom_onset_time <= self.time
            and node.contact_traced
            and not node.isolated
            and not node.completed_isolation
        ]

        # Look for houses that need to propagate the contact tracing because their test result has come back
        # Necessary conditions: household isolated, symptom onset + testing delay = time

        # Propagate the contact tracing for all households that self-reported and have had their test results come back
        [
            self.propagate_contact_tracing(node.household())
            for node in self.network.all_nodes()
            if node.time_of_reporting + node.testing_delay == self.time
            and not node.household().propagated_contact_tracing
        ]

        # Propagate the contact tracing for all households that are isolated due to exposure, have developed symptoms and had a test come back
        [
            self.propagate_contact_tracing(node.household())
            for node in self.network.all_nodes()
            if node.symptom_onset_time <= self.time
            and not node.household().propagated_contact_tracing
            and node.household().isolated_time + node.testing_delay <= self.time
        ]

        # Update the contact tracing index of households
        # That is, re-evaluate how far away they are from a known infected case (traced + symptom_onset_time + testing_delay)
        self.update_contact_tracing_index()

        if self.do_2_step:
            # Propagate the contact tracing from any households with a contact tracing index of 1
            [
                self.propagate_contact_tracing(household)
                for household in self.network.houses.all_households()
                if household.contact_tracing_index == 1
                and not household.propagated_contact_tracing
                and household.isolated
            ]

    def contact_trace_household(self, household: Household):
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
            self.network.graph.edges[edge[0], edge[1]].update({"edge_type": EdgeType.within_house.name})
            for edge in household.within_house_edges
        ]

        # If there are any nodes in the house that are symptomatic, isolate the house:
        symptomatic_nodes = [node for node in household.nodes() if node.symptom_onset_time <= self.time and not node.completed_isolation]
        if symptomatic_nodes != []:
            self.isolate_household(household)

        # work out which was the traced node
        tracing_household = self.network.houses.household(household.being_contact_traced_from)
        traced_node_id = self.network.get_edge_between_household(household, tracing_household)[0]
        traced_node = self.network.node(traced_node_id)

        # the traced node should go into quarantine
        if not traced_node.isolated and node.will_uptake_isolation:
            traced_node.isolated = True

    def perform_recoveries(self):
        """
        Loops over all nodes in the branching process and determine recoveries.

        time - The current time of the process, if a nodes recovery time equals the current time, then it is set to the recovered state
        """
        for node in self.network.all_nodes():
            if node.recovery_time == self.time:
                node.recovered = True

    def label_node_edges_between_houses(self, house_to: Household, house_from: Household, new_edge_type):
        # Annoying bit of logic to find the edge and label it
        for node_1 in house_to.nodes():
            for node_2 in house_from.nodes():
                if self.network.graph.has_edge(node_1.node_id, node_2.node_id):
                    self.network.graph.edges[node_1.node_id, node_2.node_id].update({"edge_type": new_edge_type})

    def attempt_contact_trace_of_household(self, house_to: Household, house_from: Household, contact_trace_delay: int = 0):
        # Decide if the edge was traced by the app
        app_traced = self.network.is_edge_app_traced(self.network.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self.contact_tracing_success_prob

        # is the trace successful
        if (npr.binomial(1, success_prob) == 1):
            # Update the list of traced households from this one
            house_from.contact_traced_household_ids.append(house_to.house_id)

            # Assign the household a contact tracing index, 1 more than it's parent tracer
            house_to.contact_tracing_index = house_from.contact_tracing_index + 1

            # work out the time delay
            contact_trace_delay = contact_trace_delay + self.contact_trace_delay(app_traced)
            proposed_time_until_contact_trace = self.time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed time until contact trace
            # Note this starts as infinity
            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < house_to.time_until_contact_traced:
                house_to.time_until_contact_traced = proposed_time_until_contact_trace
                house_to.being_contact_traced_from = house_from.house_id

            # Edge labelling
            if app_traced:
                self.label_node_edges_between_houses(house_to, house_from, EdgeType.app_traced.name)
            else:
                self.label_node_edges_between_houses(house_to, house_from, EdgeType.between_house.name)
        else:
            self.label_node_edges_between_houses(house_to, house_from, EdgeType.failed_contact_tracing.name)

    def isolate_household(self, household: Household):
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
            household.isolated_time = self.time

            # Update every node in the house to the isolated status
            for node in household.nodes():
                if node.will_uptake_isolation:
                    node.isolated = True

            # Which house started the contact trace that led to this house being isolated, if there is one
            # A household may be being isolated because someone in the household self reported symptoms
            # Hence sometimes there is a None value for House which contact traced
            if household.being_contact_traced_from is not None:
                house_which_contact_traced = self.network.houses.household(household.being_contact_traced_from)
                
                # Initially the edge is assigned the contact tracing label, may be updated if the contact tracing does not succeed
                if self.network.is_edge_app_traced(self.network.get_edge_between_household(household, house_which_contact_traced)):
                    self.label_node_edges_between_houses(household, house_which_contact_traced, EdgeType.app_traced.name)
                else:
                    self.label_node_edges_between_houses(household, house_which_contact_traced, EdgeType.between_house.name)
                        
                    # We update the label of every edge so that we can tell which household have been contact traced when we visualise
            [
                self.network.graph.edges[edge[0], edge[1]].update({"edge_type": EdgeType.within_house.name})
                for edge in household.within_house_edges
            ]

    def propagate_contact_tracing(self, household: Household):
        """
        To be called after a node in a household either reports their symptoms, and gets tested, when a household that is under surveillance develops symptoms + gets tested.
        """
        # update the propagation data
        household.propagated_contact_tracing = True
        household.time_propagated_tracing = self.time

        # Contact tracing attempted for the household that infected the household currently propagating the infection
        
        infected_by = household.infected_by()

        # If infected by = None, then it is the origin node, a special case
        if infected_by and not infected_by.isolated:
            self.attempt_contact_trace_of_household(infected_by, household)

        # Contact tracing for the households infected by the household currently traced
        child_households_not_traced = [h for h in household.spread_to() if not h.isolated]
        for child in child_households_not_traced:
            self.attempt_contact_trace_of_household(child, household)

    def update_contact_tracing_index(self):
        for household in self.network.houses.all_households():
            # loop over households with non-zero indexes, those that have been contact traced but with
            if household.contact_tracing_index != 0:
                for node in household.nodes():

                    # Necessary conditions for an index 1 household to propagate tracing:
                    # The node must have onset of symptoms
                    # The node households must be isolated
                    # The testing delay must be passed
                    # The testing delay starts when the house have been isolated and symptoms have onset
                    critical_time = max(node.symptom_onset_time, household.isolated_time)

                    if critical_time + node.testing_delay <= self.time:
                        household.contact_tracing_index = 0

                        for index_1_hh in household.contact_traced_households():
                            if index_1_hh.contact_tracing_index == 2:
                                index_1_hh.contact_tracing_index = 1

    def isolate_self_reporting_cases(self):
        """Applies the isolation status to nodes who have reached their self-report time.
        They may of course decide to not adhere to said isolation, or may be a member of a household
        who will not uptake isolation
        """
        for node in self.network.all_nodes():
            if node.will_uptake_isolation:
                 if node.time_of_reporting == self.time:
                    node.isolated = True

    def release_nodes_from_quarantine_or_isolation(self):
        """If a node has completed the quarantine according to the following rules, they are released from
        quarantine.

        You are released from isolation if:
            * it has been 10 days since your symptoms onset
            and
            * it has been a minimum of 14 days since your household was isolated
        """

        # We consider two distinct cases, and define logic for each case
        self.release_nodes_who_completed_isolation()
        self.release_nodes_who_completed_quarantine()

    def release_nodes_who_completed_quarantine(self):
        """If a node is currently in quarantine, and has completed the quarantine period then we release them from quarantine.

        An individual is in quarantine if they have been contact traced, and have not had symptom onset.

        A quarantined individual is released from quarantine if it has been quarantine_duration since they last had contact with a known case.
        In our model, this corresponds to the time of infection.
        """
        for node in self.network.all_nodes():
            # For nodes who do not self-report, and are in the same household as their infector
            # (if they do not self-report they will not isolate; if contact traced, they will be quarantining for the quarantine duration)          
            #if node.household_id == node.infected_by_node().household_id:
            if node.infected_by_node():
                if (node.infection_status(self.time) == "unknown_infection") & node.isolated:
                    if node.locally_infected():

                        if self.time >= (node.household().earliest_recognised_symptom_onset(model_time = self.time) + self.quarantine_duration):
                            node.isolated = False
                            node.completed_isolation = True  
                            node.completed_isolation_reason = 'completed_quarantine'
                            node.completed_isolation_time = self.time
                # For nodes who do not self-report, and are not in the same household as their infector
                # (if they do not self-report they will not isolate; if contact traced, they will be quarantining for the quarantine duration)          
                    elif node.contact_traced & (self.time >= node.time_infected + self.quarantine_duration):
                        node.isolated = False
                        node.completed_isolation = True 
                        node.completed_isolation_time = self.time
                        node.completed_isolation_reason = 'completed_quarantine'
        
    def release_nodes_who_completed_isolation(self):
        """
        Nodes leave self-isolation, rather than quarantine, when their infection status is either known (ie tested) or when they are in a 
        contact traced household and they develop symptoms (they might then go on to get a test, but they isolate regardless). Nodes in contact traced households do not have a will_report_infection probability: if they develop symptoms, they are a self-recognised infection who might or might not go on to test and become a known infection.

        If it has been isolation_duration since these individuals have had symptom onset, then they are released from isolation.
        """
        for node in self.network.all_nodes():
            if node.isolated:
                if node.infection_status(self.time)=="known_infection" or node.infection_status(self.time)=="self_recognised_infection":
                    if self.time >= node.symptom_onset_time + self.self_isolation_duration:
                        node.isolated = False
                        node.completed_isolation = True
                        node.completed_isolation_time = self.time
                        node.completed_isolation_reason = 'completed_isolation'

    def _simulate_one_step(self):
        """ Private method: Simulates one day of the epidemic and contact tracing."""
        # perform a days worth of infections
        self.increment_infection()
        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        self.isolate_self_reporting_cases()
        # isolate self-reporting-nodes while they wait for tests
        self.update_isolation()
        # propagate contact tracing
        for _ in range(5):
            self.increment_contact_tracing()
        # node recoveries
        self.perform_recoveries()
        # release nodes from quarantine or isolation if the time has arrived
        self.release_nodes_from_quarantine_or_isolation()
        # increment time
        self.time += 1


    def initialise_simulation(self):
        """ Initialise the simulation to its starting values. """

        # At step (day) zero
        self.time = 0

        # Reset the network (nodes, houses and graph)
        self.network.reset()

        # Create first household
        # Initial values
        house_id = 0
        generation = 0

        # Create the starting infectives
        for _ in range(self.starting_infections):
            house_id += 1
            node_id = self.network.node_count + 1
            self.new_household(house_id, 1, None, None)
            self.new_infection(node_id, generation, house_id)

        # Call parent initialised_simulation
        BPSimulationModel.simulation_initialised(self)


    def run_simulation(self, num_steps: int, infection_threshold: int = 100000) -> None:
        """ Runs the simulation:
                Sets model state,
                Announces start/stopped and step increments to observers

        Arguments:
            node: num steps -- The number of step increments to perform
            infection_threshold -- The maximum number of infectious nodes allowed, befure stopping stimulation

        Returns:
            None
        """

        # Tell parent simulation started
        BPSimulationModel.simulation_started(self)

        while type(self.state) is RunningState:
            prev_graph = self.network.graph.copy()

            # This chunk of code executes one step (a days worth of infections and contact tracings)
            self._simulate_one_step()

            # If graph changed, tell parent
            new_graph = self.network.graph
            if not graphs_isomorphic(prev_graph, new_graph):
                BPSimulationModel.graph_changed(self)

            # Call parent completed step
            BPSimulationModel.completed_step_increment(self)

            # Simulation ends if num_steps is reached
            if self.time == num_steps:
                self.state.timed_out()
            elif self.network.count_non_recovered_nodes() == 0:
                self.state.go_extinct()
            elif self.network.count_non_recovered_nodes() > infection_threshold:
                self.state.max_nodes_infectious()

        # Tell parent simulation stopped
        BPSimulationModel.simulation_stopped(self)

    def node_type(self, node: Node):
        """ Returns a node type, given the current status of the node.

        Arguments:
            node: Node -- The node

        Returns:
            str -- The status assigned
        """

        return node.node_type()

class uk_model(household_sim_contact_tracing):

    def __init__(self, params: dict, prob_testing_positive_pcr_func: Callable[[int], float]):

        validate_parameters(params, "./schemas/uk_model.json")

        super().__init__(params)

        self.prob_testing_positive_pcr_func = prob_testing_positive_pcr_func

        if "number_of_days_to_trace_backwards" in params:
            self.number_of_days_to_trace_backwards = params["number_of_days_to_trace_backwards"]
        else:
            self.number_of_days_to_trace_backwards = 2
        if "number_of_days_to_trace_forwards" in params:
            self.number_of_days_to_trace_forwards = params["number_of_days_to_trace_forwards"]
        else:
            self.number_of_days_to_trace_forwards = 7
        if "probable_infections_need_test" in params:
            self.probable_infections_need_test = params["probable_infections_need_test"]
        else:
            self.probable_infections_need_test = True
        if "recall_probability_fall_off" in params:
            self.recall_probability_fall_off = params["recall_probability_fall_off"]
        else:
            self.recall_probability_fall_off = 1

    def update_isolation(self):
        # Update the contact traced status for all households that have had the contact
        # tracing process get there
        [
            self.contact_trace_household(household)
            for household in self.network.houses.all_households()
            if household.time_until_contact_traced <= self.time
            and not household.contact_traced
        ]

        # Isolate all non isolated households where the infection has been reported
        # (excludes those who will not take up isolation if prob <1)
        [
            self.isolate_household(node.household())
            for node in self.network.all_nodes()
            if node.time_of_reporting + node.testing_delay == self.time
            and node.received_positive_test_result
            and not node.household().isolated
            and not node.household().contact_traced
        ]

    def pcr_test_node(self, node: Node):
        """Given the nodes infectious age, will that node test positive

        Args:
            node (Node): The node to be tested today
        """
        node.received_result = True

        infectious_age_when_tested = self.time - node.testing_delay - node.time_infected

        prob_positive_result = self.prob_testing_positive_pcr_func(infectious_age_when_tested)

        if npr.binomial(1, prob_positive_result) == 1:
            node.received_positive_test_result = True
        else:
            node.received_positive_test_result = False

    def receive_pcr_test_results(self):
        """For nodes who would receive a PCR test result today, update
        """
        # self reporting infections
        [
            self.pcr_test_node(node)
            for node in self.network.all_nodes()
            if node.time_of_reporting + node.testing_delay == self.time
            and not node.received_result
            and not node.contact_traced
        ]

        # contact traced nodes
        [
            self.pcr_test_node(node)
            for node in self.network.all_nodes()
            if node.symptom_onset_time + node.testing_delay == self.time
            and node.contact_traced
            and not node.received_result
        ]

    def increment_contact_tracing(self):

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
            self.isolate_household(node.household())
            for node in self.network.all_nodes()
            if node.symptom_onset_time <= self.time
            and node.received_positive_test_result
            and not node.isolated
            and not node.completed_isolation
        ]

        [
            self.propagate_contact_tracing(node)
            for node in self.network.all_nodes()
            if node.received_result
            and not node.propagated_contact_tracing
        ]

    def contact_trace_household(self, household: Household):
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
            self.network.graph.edges[edge[0], edge[1]].update({"edge_type": EdgeType.within_house.name})
            for edge in household.within_house_edges
        ]

        # work out which was the traced node
        tracing_household = self.network.houses.household(household.being_contact_traced_from)
        traced_node_id = self.network.get_edge_between_household(household, tracing_household)[0]
        traced_node = self.network.node(traced_node_id)

        # the traced node should go into quarantine
        if not traced_node.isolated:
            if node.will_uptake_isolation:
                traced_node.isolated = True

    def propagate_contact_tracing(self, node: Node):
        """
        To be called after a node in a household either reports their symptoms, and gets tested, when a household that is under surveillance develops symptoms + gets tested.
        """
        # update the propagation data
        node.propagated_contact_tracing = True
        node.time_propagated_tracing = self.time

        # Contact tracing attempted for the household that infected the household currently propagating the infection
        infected_by_node = node.infected_by_node()

        # If the node was globally infected, we are backwards tracing and the infecting node is not None
        if not node.locally_infected() and infected_by_node:

            # if the infector is not already isolated and the time the node was infected captured by going backwards
            # the node.time_infected is when they had a contact with their infector.
            if  not infected_by_node.isolated and node.time_infected >= node.symptom_onset_time - self.number_of_days_to_trace_backwards:

                # Then attempt to contact trace the household of the node that infected you
                self.attempt_contact_trace_of_household(
                    house_to=infected_by_node.household(),
                    house_from=node.household(),
                    days_since_contact_occurred=self.time - node.time_infected
                    )

        # spread_to_global_node_time_tuples stores a list of tuples, where the first element is the node_id
        # of a node who was globally infected by the node, and the second element is the time of transmission
        for global_infection in node.spread_to_global_node_time_tuples:

            # Get the child node_id and the time of transmission/time of contact
            child_node_id, time = global_infection

            child_node = self.network.node(child_node_id)

            # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
            if time >= node.symptom_onset_time - self.number_of_days_to_trace_backwards and time <= node.symptom_onset_time + self.number_of_days_to_trace_forwards and not child_node.isolated:

                self.attempt_contact_trace_of_household(
                    house_to=child_node.household(),
                    house_from=node.household(),
                    days_since_contact_occurred=self.time - time
                    )

    def attempt_contact_trace_of_household(self, house_to: Household, house_from: Household, days_since_contact_occurred: int, contact_trace_delay: int = 0):
        # Decide if the edge was traced by the app
        app_traced = self.network.is_edge_app_traced(self.network.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self.contact_tracing_success_prob * self.recall_probability_fall_off ** days_since_contact_occurred

        # is the trace successful
        if (npr.binomial(1, success_prob) == 1):
            # Update the list of traced households from this one
            house_from.contact_traced_household_ids.append(house_to.house_id)

            # Assign the household a contact tracing index, 1 more than it's parent tracer
            house_to.contact_tracing_index = house_from.contact_tracing_index + 1

            # work out the time delay
            proposed_time_until_contact_trace = self.time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed time until contact trace
            # Note this starts as infinity
            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < house_to.time_until_contact_traced:
                house_to.time_until_contact_traced = proposed_time_until_contact_trace
                house_to.being_contact_traced_from = house_from.house_id

            # Edge labelling
            if app_traced:
                self.label_node_edges_between_houses(house_to, house_from, EdgeType.app_traced.name)
            else:
                self.label_node_edges_between_houses(house_to, house_from, EdgeType.between_house.name)
        else:
            self.label_node_edges_between_houses(house_to, house_from, EdgeType.failed_contact_tracing.name)


class ContactModelTest(uk_model):

    def __init__(self, params, prob_testing_positive_pcr_func, prob_testing_positive_lfa_func):

        validate_parameters(params, "./schemas/contact_model_test.json")

        self.prob_testing_positive_lfa_func = prob_testing_positive_lfa_func

        self.LFA_testing_requires_confirmatory_PCR = params["LFA_testing_requires_confirmatory_PCR"]
        self.policy_for_household_contacts_of_a_positive_case = params["policy_for_household_contacts_of_a_positive_case"]

        if "number_of_days_prior_to_LFA_result_to_trace" in params:
            self.number_of_days_prior_to_LFA_result_to_trace = params["number_of_days_prior_to_LFA_result_to_trace"]
        else:
            self.number_of_days_prior_to_LFA_result_to_trace = 2
        if "propensity_risky_behaviour_lfa_testing" in params:
            self.propensity_risky_behaviour_lfa_testing = params["propensity_risky_behaviour_lfa_testing"]
        else:
            self.propensity_risky_behaviour_lfa_testing = 0
        if "global_contact_reduction_risky_behaviour" in params:
            self.global_contact_reduction_risky_behaviour = params["global_contact_reduction_risky_behaviour"]
        else:
            self.global_contact_reduction_risky_behaviour = 0
        if "node_daily_prob_lfa_test" in params:
            self.node_daily_prob_lfa_test = params["node_daily_prob_lfa_test"]
        else:
            self.node_daily_prob_lfa_test = 1
        if "proportion_with_propensity_miss_lfa_tests" in params:
            self.proportion_with_propensity_miss_lfa_tests = params["proportion_with_propensity_miss_lfa_tests"]
        else:
            self.proportion_with_propensity_miss_lfa_tests = 0
        if "node_prob_will_take_up_lfa_testing" in params:
            self.node_prob_will_take_up_lfa_testing = params["node_prob_will_take_up_lfa_testing"]
        else:
            self.node_prob_will_take_up_lfa_testing = 1
        if "lateral_flow_testing_duration" in params:
            self.lateral_flow_testing_duration = params["lateral_flow_testing_duration"]
        else:
            self.lateral_flow_testing_duration = 7
        if "lfa_tested_nodes_book_pcr_on_symptom_onset" in params:
            self.lfa_tested_nodes_book_pcr_on_symptom_onset = params["lfa_tested_nodes_book_pcr_on_symptom_onset"]
        else:
            self.lfa_tested_nodes_book_pcr_on_symptom_onset = True

        super().__init__(params, prob_testing_positive_pcr_func)

    def pcr_test_node(self, node: Node):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (Node): The node to be tested today
        """
        node.received_result = True
        
        infectious_age_when_tested = self.time - node.testing_delay - node.time_infected

        prob_positive_result = self.prob_testing_positive_pcr_func(infectious_age_when_tested)

        if npr.binomial(1, prob_positive_result) == 1:
            node.received_positive_test_result = True
            node.avenue_of_testing = 'PCR'
            node.positive_test_time = self.time
        else:
            node.received_positive_test_result = False
            node.avenue_of_testing = 'PCR'

    def lfa_test_node(self, node: Node):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (Node): The node to be tested today
        """

        infectious_age = self.time - node.time_infected

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
    
    def will_engage_in_risky_behaviour_while_being_lfa_tested(self):
        """Will the node engage in more risky behaviour if they are being LFA tested?
        """
        if npr.binomial(1, self.propensity_risky_behaviour_lfa_testing) == 1:
            return True
        else:
            return False

    def will_take_up_lfa_testing(self) -> bool:
        return npr.binomial(1, self.node_prob_will_take_up_lfa_testing) == 1

    def get_contact_rate_reduction(self, node: Node):
        """This method overides the default behaviour. Previously the overide behaviour allowed the global
        contact reduction to vary by household size.

        We override this behaviour, so that we can vary the global contact reduction by whether a node is
        isolating or being lfa tested or whether they engage in risky behaviour while they are being lfa tested.

        Remember that a contact rate reduction of 1 implies that 100% of conacts are stopped.
        """
        # the isolated status should never apply to an individual who will not uptake isolation

        if node.isolated and not node.propensity_imperfect_isolation:
            # perfect isolation
            return 1

        elif node.isolated and node.propensity_imperfect_isolation:
            # imperfect isolation
            return self.global_contact_reduction_imperfect_quarantine

        elif node.being_lateral_flow_tested and node.propensity_risky_behaviour_lfa_testing:
            # engaging in risky behaviour while testing negative
            return self.global_contact_reduction_risky_behaviour

        else:
            # normal levels of social distancing
            return self.reduce_contacts_by

    def new_household(
        self,
        new_household_number: int, 
        generation: int, 
        infected_by: int, 
        infected_by_node: int,
        additional_attributes: Optional[dict] = None):

        super().new_household(
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

    def new_infection(
        self,
        node_count: int, 
        generation: int,
        household_id: int,
        serial_interval=None,
        infecting_node: Optional[Node]=None,
        additional_attributes: Optional[dict] = None):
        """Add a new infection to the model and network. Attributes are randomly generated.

        This method passess additional attribute, relevant to the lateral flow testing.

        Args:
            node_count (int): The number of nodes already in the model
            generation (int): The generation of the node
            household_id (int): The household id that the node is being added to
            serial_interval ([type]): The serial interval
            infecting_node (Optional[Node]): The id of the infecting node
            additional_attributes (Optional[dict]): Additional attributes to be passed
        """

        household = self.network.houses.household(household_id)

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
            'propensity_risky_behaviour_lfa_testing': self.will_engage_in_risky_behaviour_while_being_lfa_tested(),
            'propensity_to_miss_lfa_tests': self.propensity_to_miss_lfa_tests()
        }

        if additional_attributes:
            # if new additional attributes are passed, these overide the current additional attributes if they are the same value
            # if they are different values, then they are added to the dictionary
            additional_attributes_with_defaults = {**default_additional_attributes, **additional_attributes}
        else:
            additional_attributes_with_defaults = default_additional_attributes

        asymptomatic = self.is_asymptomatic_infection()

        # Symptom onset time
        symptom_onset_time = self.time + self.incubation_period(asymptomatic)

        # If the node is asymptomatic, we need to generate a pseudo symptom onset time
        if asymptomatic:
            pseudo_symptom_onset_time = self.incubation_period(asymptomatic=False)
        else:
            pseudo_symptom_onset_time = symptom_onset_time

        # When a node reports its infection
        if not asymptomatic and npr.binomial(1, self.infection_reporting_prob) == 1:
            will_report_infection = True
            time_of_reporting = symptom_onset_time + self.reporting_delay(asymptomatic)
        else:
            will_report_infection = False
            time_of_reporting = float('Inf')

        # We assign each node a recovery period of 21 days, after 21 days the probability of
        # causing a new infections is 0, due to the generation time distribution
        recovery_time = self.time + 14

        household = self.network.houses.household(household_id)

        # If the household has the propensity to use the contact tracing app, decide
        # if the node uses the app.
        if household.propensity_trace_app:
            has_trace_app = self.has_contact_tracing_app()
        else:
            has_trace_app = False

        # in case you want to add non-default additional attributes
        default_additional_attributes = {}

        if additional_attributes_with_defaults:
            default_additional_attributes = {**default_additional_attributes, **additional_attributes_with_defaults}

        isolation_uptake = self.will_uptake_isolation()

        if household.isolated and isolation_uptake:
            node_is_isolated = True
        else:
            node_is_isolated = False

        self.network.add_node(
            node_id=node_count,
            time=self.time,
            generation=generation,
            household=household_id,
            isolated=node_is_isolated,
            will_uptake_isolation=isolation_uptake,
            propensity_imperfect_isolation=self.get_propensity_imperfect_isolation(),
            asymptomatic=asymptomatic,
            contact_traced=household.contact_traced,
            symptom_onset_time=symptom_onset_time,
            pseudo_symptom_onset_time=pseudo_symptom_onset_time,
            serial_interval=serial_interval,
            recovery_time=recovery_time,
            will_report_infection=will_report_infection,
            time_of_reporting=time_of_reporting,
            has_contact_tracing_app=has_trace_app,
            testing_delay=self.testing_delay(),
            additional_attributes=default_additional_attributes,
            infecting_node=infecting_node,
        )

        # Each house now stores the ID's of which nodes are stored inside the house,
        # so that quarantining can be done at the household level
        household.node_ids.append(node_count)
    
    def will_take_up_lfa_testing(self) -> bool:
        return npr.binomial(1, self.node_prob_will_take_up_lfa_testing) == 1

    def propensity_to_miss_lfa_tests(self) -> bool:
        return npr.binomial(1, self.proportion_with_propensity_miss_lfa_tests) == 1

    def contact_trace_household(self, household: Household):
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
            self.network.graph.edges[edge[0], edge[1]].update({"edge_type": EdgeType.within_house.name})
            for edge in household.within_house_edges
        ]

        # work out which was the traced node
        tracing_household = self.network.houses.household(household.being_contact_traced_from)
        traced_node_id = self.network.get_edge_between_household(household, tracing_household)[0]
        traced_node = self.network.node(traced_node_id)

        # the traced node is now being lateral flow tested
        if traced_node.node_will_take_up_lfa_testing and not traced_node.received_positive_test_result:
            traced_node.being_lateral_flow_tested = True
            traced_node.time_started_lfa_testing = self.time

    def propagate_contact_tracing(self, node: Node):
        """
        To be called after a node in a household either reports their symptoms, and gets tested, when a household that is under surveillance develops symptoms + gets tested.
        """

        # TODO: Refactor this monster
        # There are really 3 contact tracing algorithms going on here
        # 1) Trace on non-confirmatory PCR result
        # 2) Trace on confirmatory PCR result

        # update the propagation data
        node.propagated_contact_tracing = True
        node.time_propagated_tracing = self.time

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
                        days_since_contact_occurred=self.time - node.time_infected
                        )

            elif node.avenue_of_testing == 'LFA':

                if not self.LFA_testing_requires_confirmatory_PCR:

                    if not infected_by_node.isolated and node.time_infected >= node.positive_test_time - self.number_of_days_prior_to_LFA_result_to_trace:

                        # Then attempt to contact trace the household of the node that infected you
                        self.attempt_contact_trace_of_household(
                            house_to=infected_by_node.household(),
                            house_from=node.household(),
                            days_since_contact_occurred=self.time - node.time_infected
                            )

        # spread_to_global_node_time_tuples stores a list of tuples, where the first element is the node_id
        # of a node who was globally infected by the node, and the second element is the time of transmission
        for global_infection in node.spread_to_global_node_time_tuples:
            
            # Get the child node_id and the time of transmission/time of contact
            child_node_id, time = global_infection

            child_node = self.network.node(child_node_id)

            if node.avenue_of_testing == 'PCR':

                # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
                if time >= node.symptom_onset_time - self.number_of_days_to_trace_backwards and time <= node.symptom_onset_time + self.number_of_days_to_trace_forwards and not child_node.isolated:

                    self.attempt_contact_trace_of_household(
                        house_to=child_node.household(),
                        house_from=node.household(),
                        days_since_contact_occurred=self.time - time
                        )
            
            elif node.avenue_of_testing == 'LFA':

                if not self.LFA_testing_requires_confirmatory_PCR:

                    # If the node was infected 2 days prior to symptom onset, or 7 days post and is not already isolated
                    if time >= node.positive_test_time - self.number_of_days_prior_to_LFA_result_to_trace:

                        self.attempt_contact_trace_of_household(
                            house_to=child_node.household(),
                            house_from=node.household(),
                            days_since_contact_occurred=self.time - time
                            )

    def start_lateral_flow_testing_household(self, household: Household):
        """Sets the household to the lateral flow testing status so that new within household infections are tested

        All nodes in the household start lateral flow testing

        Args:
            household (Household): The household which is initiating testing
        """

        household.being_lateral_flow_tested = True
        household.being_lateral_flow_tested_start_time = self.time

        for node in household.nodes():
            if node.node_will_take_up_lfa_testing and not node.received_positive_test_result and not node.being_lateral_flow_tested: 
                node.being_lateral_flow_tested = True
                node.time_started_lfa_testing = self.time

    def start_lateral_flow_testing_household_and_quarantine(self, household: Household):
        """Sets the household to the lateral flow testing status so that new within household infections are tested

        All nodes in the household start lateral flow testing and start quarantining

        Args:
            household (Household): The household which is initiating testing
        """
        household.being_lateral_flow_tested = True
        household.being_lateral_flow_tested_start_time = self.time
        household.isolated = True
        household.isolated_time = True
        household.contact_traced = True

        for node in household.nodes():
            if node.node_will_take_up_lfa_testing and not node.received_positive_test_result and not node.being_lateral_flow_tested: 
                node.being_lateral_flow_tested = True
                node.time_started_lfa_testing = self.time

            if node.will_uptake_isolation:
                node.isolated = True

    def start_household_quarantine(self, household: Household):
        """Sets the household to the lateral flow testing status so that new within household infections are tested

        All nodes in the household start lateral flow testing

        Args:
            household (Household): The household which is initiating testing
        """
        self.isolate_household(household)

    def apply_policy_for_household_contacts_of_a_positive_case(self, household: Household):
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
            self.start_lateral_flow_testing_household(household)
        elif self.policy_for_household_contacts_of_a_positive_case == 'lfa testing and quarantine':
            self.start_lateral_flow_testing_household_and_quarantine(household)
        elif self.policy_for_household_contacts_of_a_positive_case == 'no lfa testing only quarantine':
            self.start_household_quarantine(household)
        else:
            raise Exception("""policy_for_household_contacts_of_a_positive_case not recognised. Must be one of the following options:
                * "lfa testing no quarantine"
                * "lfa testing and quarantine"
                * "no lfa testing only quarantine" """)

    def act_on_confirmatory_pcr_results(self):
        """Once on a individual receives a positive pcr result we need to act on it.

        This takes the form of:
        * Household members start lateral flow testing
        * Contact tracing is propagated
        """
        
        [
            self.apply_policy_for_household_contacts_of_a_positive_case(node.household())
            for node in self.network.all_nodes()
            if node.confirmatory_PCR_test_result_time == self.time
        ]

    def get_positive_lateral_flow_nodes(self):
        """Performs a days worth of lateral flow testing.

        Returns:
            List[Nodes]: A list of nodes who have tested positive through the lateral flow tests. 
        """

        return [
            node for node in self.network.all_nodes()
            if node.being_lateral_flow_tested
            and self.will_lfa_test_today(node)
            and not node.received_positive_test_result
            and self.lfa_test_node(node)
        ]

    def isolate_positive_lateral_flow_tests(self):
        """A if a node tests positive on LFA, we assume that they isolate and stop LFA testing

        If confirmatory PCR testing is not required, then we do not start LFA testing the household at this point in time.
        """

        for node in self.current_LFA_positive_nodes:
            node.received_positive_test_result = True

            if node.will_uptake_isolation:
                node.isolated = True

            node.avenue_of_testing = 'LFA'
            node.positive_test_time = self.time
            node.being_lateral_flow_tested = False

            if not node.household().applied_policy_for_household_contacts_of_a_positive_case and not self.LFA_testing_requires_confirmatory_PCR:

                self.apply_policy_for_household_contacts_of_a_positive_case(node.household())

    def take_confirmatory_pcr_test(self, node: Node):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (Node): The node to be tested today
        """
        
        infectious_age_when_tested = self.time - node.time_infected
        prob_positive_result = self.prob_testing_positive_pcr_func(infectious_age_when_tested)

        node.confirmatory_PCR_test_time = self.time
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

    def act_on_positive_LFA_tests(self):
        """For nodes who test positive on their LFA test, take the appropriate action depending on the policy
        """
        self.current_LFA_positive_nodes = self.get_positive_lateral_flow_nodes()

        self.isolate_positive_lateral_flow_tests()

        if self.LFA_testing_requires_confirmatory_PCR:
            self.confirmatory_pcr_test_LFA_nodes()

    def receive_pcr_test_results(self):
        """
        For nodes who would receive a PCR test result today, update
        """

        if self.lfa_tested_nodes_book_pcr_on_symptom_onset:

            # self reporting infections who have not been contact traced
            [
                self.pcr_test_node(node)
                for node in self.network.all_nodes()
                if node.time_of_reporting + node.testing_delay == self.time
                and not node.received_result
                and not node.contact_traced
            ]

            # contact traced nodes should book a pcr test if they develop symptoms
            # we assume that this occurs at symptom onset time since they are traced
            # and on the lookout for developing symptoms
            [
                self.pcr_test_node(node)
                for node in self.network.all_nodes()
                if node.symptom_onset_time + node.testing_delay == self.time
                and not node.received_result
                and node.contact_traced
            ]

        else:

            [
                self.pcr_test_node(node)
                for node in self.network.all_nodes()
                if node.time_of_reporting + node.testing_delay == self.time
                and not node.received_result
                and not node.contact_traced
                and not node.being_lateral_flow_tested
            ]

    def update_isolation(self):

        # Update the contact traced status for all households that have had the contact tracing process get there
        [
            self.contact_trace_household(household)  
            for household in self.network.houses.all_households()
            if household.time_until_contact_traced <= self.time
            and not household.contact_traced
        ]

        # Isolate all non isolated households where the infection has been reported (excludes those who will not take up isolation if prob <1)
        new_pcr_test_results = [
            node for node in self.network.all_nodes()
            if node.positive_test_time == self.time
            and node.avenue_of_testing == 'PCR'
            and node.received_positive_test_result
        ]

        [
            self.apply_policy_for_household_contacts_of_a_positive_case(node.household())
            for node in new_pcr_test_results
            if not node.household().applied_policy_for_household_contacts_of_a_positive_case
        ]

    def increment_contact_tracing(self):
        [
            self.propagate_contact_tracing(node)
            for node in self.network.all_nodes()
            if node.received_positive_test_result
            and node.avenue_of_testing == 'PCR'
            and not node.propagated_contact_tracing
        ]

        if not self.LFA_testing_requires_confirmatory_PCR:
            [
                self.propagate_contact_tracing(node)
                for node in self.network.all_nodes()
                if node.received_positive_test_result
                and node.avenue_of_testing == 'LFA'
                and not node.propagated_contact_tracing
            ]

        elif self.LFA_testing_requires_confirmatory_PCR:
            [
                self.propagate_contact_tracing(node)
                for node in self.network.all_nodes()
                if node.confirmatory_PCR_test_result_time == self.time
                and node.confirmatory_PCR_result_was_positive
                and node.avenue_of_testing == 'LFA'
                and not node.propagated_contact_tracing
            ]

    def infection_status(self, time_now: int) -> str:
        if self.contact_traced:
            if self.positive_test_time <= time_now:
                return "known_infection"
            if self.symptom_onset_time <= time_now:
                return "self_reported_infection"     
        
        else:
            if self.positive_test_time <= time_now:
                return "known_infection"
            if self.time_of_reporting <= time_now:
                return "self_recognised_infection"

        return "unknown_infection"   
        
    def earliest_recognised_symptom_onset_or_lateral_flow_test(self, model_time: int):
        """
        Return infinite if no node in household has recognised symptom onset
        """
        recognised_symptom_onsets = [
            household_node.symptom_onset_time
            for household_node in self.network.nodes()
            if household_node.infection_status(model_time) in ("known_infection", "self_recognised_infection")
        ]

        positive_test_times = [
            household_node.positive_test_time
            for household_node in self.network.nodes()
            if household_node.infection_status(model_time) in ("known_infection")
        ]

        recognised_symptom_and_positive_test_times = recognised_symptom_onsets + positive_test_times

        if recognised_symptom_and_positive_test_times != []:
            return min(recognised_symptom_and_positive_test_times)
        else:
            return float('inf')        

    def release_nodes_from_lateral_flow_testing_or_isolation(self):
            """If a node has completed the quarantine according to the following rules, they are released from
            quarantine.

            You are released from isolation if:
                * it has been 10 days since your symptoms onset (Changed from 7 to reflect updated policy, Nov 2020)
            You are released form lateral flow testing if you have reached the end of the lateral flow testing period and not yet been removed because you are positive 

            """

            # We consider two distinct cases, and define logic for each case
            self.release_nodes_who_completed_isolation()
            self.release_nodes_who_completed_lateral_flow_testing()

    def release_nodes_who_completed_isolation(self):
        """
        Nodes leave self-isolation, rather than quarantine, when their infection status is either known (ie tested) or when they are in a 
        contact traced household and they develop symptoms (they might then go on to get a test, but they isolate regardless). Nodes in contact traced households do not have a will_report_infection probability: if they develop symptoms, they are a self-recognised infection who might or might not go on to test and become a known infection.

        If it has been isolation_duration since these individuals have had symptom onset, then they are released from isolation.
        """
        for node in self.network.all_nodes():
            if node.isolated:
                if node.infection_status(self.time)=="known_infection" or node.infection_status(self.time)=="self_recognised_infection":
                    if node.avenue_of_testing == "LFA":
                        if self.time >= node.positive_test_time + self.self_isolation_duration:
                            node.isolated = False
                            node.completed_isolation = True
                            node.completed_isolation_time = self.time 
                            node.completed_isolation_reason = 'completed_isolation'    
                    else:    
                        if self.time >= node.symptom_onset_time + self.self_isolation_duration: #this won't include nodes who tested positive due to LF tests who do not have symptoms
                            node.isolated = False
                            node.completed_isolation = True
                            node.completed_isolation_time = self.time
                            node.completed_isolation_reason = 'completed_isolation' 

    def release_nodes_who_completed_lateral_flow_testing(self):
        """If a node is currently in lateral flow testing, and has completed this period then we release them from testing.

        An individual is in lateral flow testing if they have been contact traced, and have not had symptom onset.

        They continue to be lateral flow tested until the duration of this period is up OR they test positive on lateral flow and they are isolated and traced.

        A lateral flow tested individual is released from testing if it has been 'lateral_flow_testing_duration' since they last had contact with a known case.
        In our model, this corresponds to the time of infection.
        """


        for node in self.network.all_nodes():
            if self.time >= node.time_started_lfa_testing + self.lateral_flow_testing_duration and node.being_lateral_flow_tested:
                node.being_lateral_flow_tested = False
                node.completed_lateral_flow_testing_time = self.time

        # for node in self.network.all_nodes():

        #     # For nodes who do not self-report, and are in the same household as their infector
        #     # (if they do not self-report they will not isolate; if contact traced, they will be lateral flow testing for the lateral_flow_testing_duration unless they test positive)          
        #     #if node.household_id == node.infected_by_node().household_id:
        #     if node.infected_by_node():
        #         #if (node.infection_status(self.time) == "unknown_infection") & node.being_lateral_flow_tested:
        #         if node.being_lateral_flow_tested:
        #             if node.locally_infected():

        #                 if self.time >= (node.household().earliest_recognised_symptom_onset_or_lateral_flow_test(model_time = self.time) + self.lateral_flow_testing_duration):
        #                     node.being_lateral_flow_tested = False
        #                     node.completed_lateral_flow_testing_time = self.time

        #         # For nodes who do not self-report, and are not in the same household as their infector
        #         # (if they do not self-report they will not isolate; if contact traced, they will be lateral flow testing for the lateral_flow_testing_duration unless they test positive)          
        #             elif node.contact_traced & (self.time >= node.time_infected + self.lateral_flow_testing_duration):
        #                 node.being_lateral_flow_tested = False
        #                 node.completed_lateral_flow_testing_time = self.time

    def simulate_one_step(self):
        """Simulates one day of the epidemic and contact tracing.

        Useful for bug testing and visualisation.
        """

        prev_graph = self.network.graph.copy()

        self.receive_pcr_test_results()
        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        self.isolate_self_reporting_cases()
        # isolate self-reporting-nodes while they wait for tests
        self.update_isolation()
        # isolate self reporting nodes
        self.act_on_positive_LFA_tests()
        # if we require PCR tests, to confirm infection we act on those
        if self.LFA_testing_requires_confirmatory_PCR:
            self.act_on_confirmatory_pcr_results()
        # perform a days worth of infections
        self.increment_infection()
        # propagate contact tracing
        for _ in range(5):
            self.increment_contact_tracing()
        # node recoveries
        self.perform_recoveries()
        # release nodes from quarantine or isolation if the time has arrived
        self.release_nodes_from_lateral_flow_testing_or_isolation()
        # increment time
        self.time += 1

        new_graph = self.network.graph

        if not self.network.graphs_isophomorphic(prev_graph, new_graph):
            BPSimulationModel.graph_changed(self)

        # Call parent simulate_one_step
        BPSimulationModel.completed_step_increment(self)

    def node_type(self, node: Node):
        """Returns a node type, given the current status of the node.

        Arguments:
            node: Node -- The node

        Returns:
            str -- The status assigned
        """

        return node.node_type_detailed()

