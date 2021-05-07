from typing import List, Optional, Callable
import numpy.random as npr
import os

from household_contact_tracing.network import Network, NetworkContractModel, Household, Node, \
    NodeContactModel, EdgeType, graphs_isomorphic
from household_contact_tracing.bp_simulation_model import BPSimulationModel
from household_contact_tracing.parameters import validate_parameters
from household_contact_tracing.simulation_states import RunningState
from household_contact_tracing.infection import Infection
from household_contact_tracing.contact_tracing import ContactTracing, ContactTraceHouseholdBP, \
    ContactTraceHouseholdUK, ContactTraceHouseholdContactModelTest


class household_sim_contact_tracing(BPSimulationModel):

    def __init__(self, params: dict):

        """Initializes a household branching process epidemic. Various contact tracing strategies can be utilized
        in an attempt to control the epidemic.

        Args:
            params (dict): A dictionary of parameters that are used in the model.
        """

        # Call parent init
        BPSimulationModel.__init__(self)

        # Parse parameters against schema to check they are valid
        validate_parameters(params, os.path.join(self.ROOT_DIR, "schemas/household_sim_contact_tracing.json"))

        self._network = self.instantiate_network()

        self._infection = Infection(self._network, params)
        self.infection.new_household = self.new_household
        self.infection.new_infection = self.new_infection
        self.infection.contact_rate_reduction = self.get_contact_rate_reduction

        self._contact_tracing = ContactTracing(self._network, params)
        self.contact_tracing.contact_trace_household = self.instantiate_contact_trace_household()
        self.contact_tracing.increment = self.increment_contact_tracing

        # isolation or quarantine parameters
        if "quarantine_duration" in params:
            self.quarantine_duration = params["quarantine_duration"]
        else:
            self.quarantine_duration = 14
        if "self_isolation_duration" in params:
            self.self_isolation_duration = params["self_isolation_duration"]
        else:
            self.self_isolation_duration = 7

        # The simulation timer
        self.time = 0

        # Calls the simulation reset function, which creates all the required dictionaries
        self.initialise_simulation()

    @property
    def network(self):
        return self._network

    @property
    def infection(self) -> Infection:
        return self._infection

    @infection.setter
    def infection(self, infection: Infection):
        self._infection = infection

    @property
    def contact_tracing(self) -> ContactTracing:
        return self._contact_tracing

    @contact_tracing.setter
    def contact_tracing(self, contact_tracing: ContactTracing):
        self._contact_tracing = contact_tracing

    def instantiate_contact_trace_household(self) -> ContactTraceHouseholdBP:
        return ContactTraceHouseholdBP(self.network)

    def instantiate_network(self) -> Network:
        return Network()

    def contact_trace_delay(self, app_traced_edge) -> int:
        if app_traced_edge:
            return 0
        else:
            return round(self.contact_trace_delay)

    def new_infection(self, node_count: int, generation: int, household_id: int,
                      serial_interval=None, infecting_node: Optional[Node] = None,
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
        asymptomatic = self.infection.is_asymptomatic_infection()

        # Symptom onset time
        symptom_onset_time = self.time + self.infection.incubation_period(asymptomatic)

        # If the node is asymptomatic, we need to generate a pseudo symptom onset time
        if asymptomatic:
            pseudo_symptom_onset_time = self.infection.incubation_period(asymptomatic=False)
        else:
            pseudo_symptom_onset_time = symptom_onset_time

        # When a node reports its infection
        if not asymptomatic and npr.binomial(1, self.infection.infection_reporting_prob) == 1:
            will_report_infection = True
            time_of_reporting = symptom_onset_time + self.infection.reporting_delay(asymptomatic)
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
            has_trace_app = self.contact_tracing.has_contact_tracing_app()
        else:
            has_trace_app = False

        # in case you want to add non-default additional attributes
        default_additional_attributes = {}

        if additional_attributes:
            default_additional_attributes = {**default_additional_attributes, **additional_attributes}

        isolation_uptake = self.infection.will_uptake_isolation()

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
            propensity_imperfect_isolation=self.infection.get_propensity_imperfect_isolation(),
            asymptomatic=asymptomatic,
            contact_traced=household.contact_traced,
            symptom_onset_time=symptom_onset_time,
            pseudo_symptom_onset_time=pseudo_symptom_onset_time,
            serial_interval=serial_interval,
            recovery_time=recovery_time,
            will_report_infection=will_report_infection,
            time_of_reporting=time_of_reporting,
            has_contact_tracing_app=has_trace_app,
            testing_delay=self.contact_tracing.testing_delay(),
            additional_attributes=default_additional_attributes,
            infecting_node=infecting_node,
        )

        # Each house now stores the ID's of which nodes are stored inside the house,
        # so that quarantining can be done at the household level
        household.node_ids.append(node_count)

    def get_contact_rate_reduction(self, node):
        """Returns a contact rate reduction, depending upon a nodes current status and various
        isolation parameters
        """

        if node.isolated and node.propensity_imperfect_isolation:
            return self.infection.global_contact_reduction_imperfect_quarantine
        elif node.isolated and not node.propensity_imperfect_isolation:
            # return 1 means 100% of contacts are stopped
            return 1
        else:
            return self.infection.reduce_contacts_by

    def new_household(self, new_household_number: int, generation: int, infected_by: int,
                      infected_by_node: int, additional_attributes: Optional[dict] = None):
        """Adds a new household to the household dictionary

        Arguments:
            new_household_number {int} -- The house id
            generation {int} -- The household generation of this household
            infected_by {int} -- Which household spread the infection to this household
            infected_by_node {int} -- Which node spread the infection to this household
        """
        house_size = self.infection.size_of_household()

        propensity_trace_app = self.contact_tracing.hh_propensity_use_trace_app()

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


    def update_isolation(self):
        # Update the contact traced status for all households that have had the contact tracing process get there
        [
            self.contact_tracing.contact_trace_household.contact_trace_household(household, self.time)
            for household in self.network.houses.all_households()
            if household.time_until_contact_traced <= self.time
            and not household.contact_traced
        ]

        # Isolate all non isolated households where the infection has been reported (excludes those who will not take up isolation if prob <1)
        [
            self.contact_tracing.contact_trace_household.isolate_household(node.household(), self.time)
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
            self.contact_tracing.contact_trace_household.isolate_household(node.household(), self.time)
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

        if self.contact_tracing.do_2_step:
            # Propagate the contact tracing from any households with a contact tracing index of 1
            [
                self.propagate_contact_tracing(household)
                for household in self.network.houses.all_households()
                if household.contact_tracing_index == 1
                and not household.propagated_contact_tracing
                and household.isolated
            ]

    def perform_recoveries(self):
        """
        Loops over all nodes in the branching process and determine recoveries.

        time - The current time of the process, if a nodes recovery time equals the current time, then it is set to the recovered state
        """
        for node in self.network.all_nodes():
            if node.recovery_time == self.time:
                node.recovered = True

    def attempt_contact_trace_of_household(self, house_to: Household, house_from: Household, contact_trace_delay: int = 0):
        # Decide if the edge was traced by the app
        app_traced = self.network.is_edge_app_traced(self.network.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self.contact_tracing.contact_tracing_success_prob

        # is the trace successful
        if (npr.binomial(1, success_prob) == 1):
            # Update the list of traced households from this one
            house_from.contact_traced_household_ids.append(house_to.house_id)

            # Assign the household a contact tracing index, 1 more than it's parent tracer
            house_to.contact_tracing_index = house_from.contact_tracing_index + 1

            # work out the time delay
            contact_trace_delay = contact_trace_delay + self.contact_tracing.contact_trace_delay
            proposed_time_until_contact_trace = self.time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed time until contact trace
            # Note this starts as infinity
            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < house_to.time_until_contact_traced:
                house_to.time_until_contact_traced = proposed_time_until_contact_trace
                house_to.being_contact_traced_from = house_from.house_id

            # Edge labelling
            if app_traced:
                self.contact_tracing.contact_trace_household.label_node_edges_between_houses(house_to, house_from, EdgeType.app_traced.name)
            else:
                self.contact_tracing.contact_trace_household.label_node_edges_between_houses(house_to, house_from, EdgeType.between_house.name)
        else:
            self.contact_tracing.contact_trace_household.label_node_edges_between_houses(house_to, house_from, EdgeType.failed_contact_tracing.name)


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
                if node.infection_status(self.time)=="known_infection" or node.infection_status(self.time)=="self_recognised_infection":
                    if self.time >= node.symptom_onset_time + self.self_isolation_duration:
                        node.isolated = False
                        node.completed_isolation = True
                        node.completed_isolation_time = self.time
                        node.completed_isolation_reason = 'completed_isolation'

    def simulate_one_step(self):
        """ Private method: Simulates one day of the epidemic and contact tracing."""
        # perform a days worth of infections
        self.infection.increment(self.time)
        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        self.isolate_self_reporting_cases()
        # isolate self-reporting-nodes while they wait for tests
        self.update_isolation()
        # propagate contact tracing
        for _ in range(5):
            self.contact_tracing.increment()
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
        self.infection.reset()

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
            self.simulate_one_step()

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


class uk_model(household_sim_contact_tracing):

    def __init__(self, params: dict, prob_testing_positive_pcr_func: Callable[[int], float]):

        validate_parameters(params, os.path.join(self.ROOT_DIR, "./schemas/uk_model.json"))

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

    def instantiate_contact_trace_household(self) -> ContactTraceHouseholdUK:
        return ContactTraceHouseholdUK(self.network)

    def update_isolation(self):
        # Update the contact traced status for all households that have had the contact
        # tracing process get there
        [
            self.contact_tracing.contact_trace_household.contact_trace_household(household, self.time)
            for household in self.network.houses.all_households()
            if household.time_until_contact_traced <= self.time
            and not household.contact_traced
        ]

        # Isolate all non isolated households where the infection has been reported
        # (excludes those who will not take up isolation if prob <1)
        [
            self.contact_tracing.contact_trace_household.isolate_household(node.household(), self.time)
            for node in self.network.all_nodes()
            if node.time_of_reporting + node.testing_delay == self.time
            and node.received_positive_test_result
            and not node.household().isolated
            and not node.household().contact_traced
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
            self.contact_tracing.contact_trace_household.isolate_household(node.household(), self.time)
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

    def propagate_contact_tracing(self, node: Node):
        """
        To be called after a node in a household either reports their symptoms, and gets tested, when a household that
        is under surveillance develops symptoms + gets tested.
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
            if  not infected_by_node.isolated and node.time_infected >= node.symptom_onset_time - \
                    self.number_of_days_to_trace_backwards:

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
            if time >= node.symptom_onset_time - self.number_of_days_to_trace_backwards and \
                    time <= node.symptom_onset_time + self.number_of_days_to_trace_forwards and not child_node.isolated:

                self.attempt_contact_trace_of_household(
                    house_to=child_node.household(),
                    house_from=node.household(),
                    days_since_contact_occurred=self.time - time
                    )

    def attempt_contact_trace_of_household(self, house_to: Household, house_from: Household,
                                           days_since_contact_occurred: int, contact_trace_delay: int = 0):
        # Decide if the edge was traced by the app
        app_traced = self.network.is_edge_app_traced(self.network.get_edge_between_household(house_from, house_to))

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
                self.contact_tracing.contact_trace_household.label_node_edges_between_houses(house_to, house_from, EdgeType.app_traced.name)
            else:
                self.contact_tracing.contact_trace_household.label_node_edges_between_houses(house_to, house_from, EdgeType.between_house.name)
        else:
            self.contact_tracing.contact_trace_household.label_node_edges_between_houses(house_to, house_from, EdgeType.failed_contact_tracing.name)


class ContactModelTest(uk_model):

    def __init__(self, params, prob_testing_positive_pcr_func, prob_testing_positive_lfa_func):

        validate_parameters(params, os.path.join(self.ROOT_DIR, "schemas/household_sim_contact_tracing.json"))

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

        if "global_contact_reduction_risky_behaviour" in params:
            self.infection.global_contact_reduction_risky_behaviour = params["global_contact_reduction_risky_behaviour"]
        else:
            self.infection.global_contact_reduction_risky_behaviour = 0

    def instantiate_contact_trace_household(self) -> ContactTraceHouseholdContactModelTest:
        return ContactTraceHouseholdContactModelTest(self.network)

    def instantiate_network(self):
        return NetworkContractModel()

    def pcr_test_node(self, node: NodeContactModel):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (NodeContactModel): The node to be tested today
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

    def lfa_test_node(self, node: NodeContactModel):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (NodeContactModel): The node to be tested today
        """

        infectious_age = self.time - node.time_infected

        prob_positive_result = self.prob_testing_positive_lfa_func(infectious_age)

        if npr.binomial(1, prob_positive_result) == 1:
            return True
        else:
            return False

    def will_lfa_test_today(self, node: NodeContactModel) -> bool:

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

    def get_contact_rate_reduction(self, node):
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
            return self.infection.global_contact_reduction_imperfect_quarantine

        elif node.being_lateral_flow_tested and node.propensity_risky_behaviour_lfa_testing:
            # engaging in risky behaviour while testing negative
            return self.infection.global_contact_reduction_risky_behaviour

        else:
            # normal levels of social distancing
            return self.infection.reduce_contacts_by


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
        infecting_node: Optional[NodeContactModel]=None,
        additional_attributes: Optional[dict] = None):
        """Add a new infection to the model and network. Attributes are randomly generated.

        This method passess additional attribute, relevant to the lateral flow testing.

        Args:
            node_count (int): The number of nodes already in the model
            generation (int): The generation of the node
            household_id (int): The household id that the node is being added to
            serial_interval ([type]): The serial interval
            infecting_node (Optional[NodeContactModel]): The id of the infecting node
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

        asymptomatic = self.infection.is_asymptomatic_infection()

        # Symptom onset time
        symptom_onset_time = self.time + self.infection.incubation_period(asymptomatic)

        # If the node is asymptomatic, we need to generate a pseudo symptom onset time
        if asymptomatic:
            pseudo_symptom_onset_time = self.infection.incubation_period(asymptomatic=False)
        else:
            pseudo_symptom_onset_time = symptom_onset_time

        # When a node reports its infection
        if not asymptomatic and npr.binomial(1, self.infection.infection_reporting_prob) == 1:
            will_report_infection = True
            time_of_reporting = symptom_onset_time + self.infection.reporting_delay(asymptomatic)
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
            has_trace_app = self.contact_tracing.has_contact_tracing_app()
        else:
            has_trace_app = False

        # in case you want to add non-default additional attributes
        default_additional_attributes = {}

        if additional_attributes_with_defaults:
            default_additional_attributes = {**default_additional_attributes, **additional_attributes_with_defaults}

        isolation_uptake = self.infection.will_uptake_isolation()

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
            propensity_imperfect_isolation=self.infection.get_propensity_imperfect_isolation(),
            asymptomatic=asymptomatic,
            contact_traced=household.contact_traced,
            symptom_onset_time=symptom_onset_time,
            pseudo_symptom_onset_time=pseudo_symptom_onset_time,
            serial_interval=serial_interval,
            recovery_time=recovery_time,
            will_report_infection=will_report_infection,
            time_of_reporting=time_of_reporting,
            has_contact_tracing_app=has_trace_app,
            testing_delay=self.contact_tracing.testing_delay(),
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

    def propagate_contact_tracing(self, node: NodeContactModel):
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
        self.contact_tracing.contact_trace_household.isolate_household(household, self.time)

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

    def take_confirmatory_pcr_test(self, node: NodeContactModel):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (NodeContactModel): The node to be tested today
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
            self.contact_tracing.contact_trace_household.contact_trace_household(household, self.time)
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

    def increment_contact_tracing(self, time):

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
            self.contact_tracing.contact_trace_household.isolate_household(node.household(), self.time)
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
        self.infection.increment(self.time)
        # propagate contact tracing
        for _ in range(5):
            self.contact_tracing.increment()
        # node recoveries
        self.perform_recoveries()
        # release nodes from quarantine or isolation if the time has arrived
        self.release_nodes_from_lateral_flow_testing_or_isolation()
        # increment time
        self.time += 1

        new_graph = self.network.graph

        if not graphs_isomorphic(prev_graph, new_graph):
            BPSimulationModel.graph_changed(self)

        # Inform parent model that step is completed
        BPSimulationModel.completed_step_increment(self)
