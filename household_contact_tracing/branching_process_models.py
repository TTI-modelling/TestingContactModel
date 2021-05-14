from typing import List, Callable
import numpy.random as npr
import os

from household_contact_tracing.network import Network, NetworkContactModel, Household, Node, \
    graphs_isomorphic, InfectionStatus, TestType
from household_contact_tracing.simulation_model import SimulationModel
from household_contact_tracing.parameters import validate_parameters
from household_contact_tracing.simulation_states import RunningState
from household_contact_tracing.infection import Infection, \
    NewHousehold, NewHouseholdContactModelTest, \
    NewInfectionHousehold, NewInfectionContactModelTest, \
    ContactRateReductionHousehold, ContactRateReductionContactModelTest
from household_contact_tracing.contact_tracing import ContactTracing, \
    ContactTraceHousehold, ContactTraceHouseholdUK, ContactTraceHouseholdContactModelTest, \
    IncrementContactTracingHousehold, IncrementContactTracingUK, IncrementContactTracingContactModelTest, \
    UpdateIsolationHousehold, UpdateIsolationUK, \
    PCRTestingUK, PCRTestingContactModelTest


class HouseholdLevelContactTracing(SimulationModel):

    def __init__(self, params: dict):

        """Initializes a household branching process epidemic. Various contact tracing strategies can be utilized
        in an attempt to control the epidemic.

        Args:
            params (dict): A dictionary of parameters that are used in the model.
        """

        # Call parent init
        SimulationModel.__init__(self)

        # Parse parameters against schema to check they are valid
        validate_parameters(params, os.path.join(self.ROOT_DIR, "schemas/household_sim_contact_tracing.json"))

        # Set default parameters
        # isolation or quarantine parameters
        self.quarantine_duration = 14
        self.self_isolation_duration = 7

        # Overwrites default params with new params if present
        for param_name in self.__dict__:
            if param_name in params:
                self.__dict__[param_name] = params[param_name]

        # Set network
        self._network = self.instantiate_network()

        # Set strategies (Strategy pattern)
        self._infection = Infection(self._network,
                                    self.instantiate_new_household(),
                                    self.instantiate_new_infection(),
                                    self.instantiate_contact_rate_reduction(),
                                    params)

        self._contact_tracing = ContactTracing(self._network,
                                               self.instantiate_contact_trace_household(),
                                               self.instantiate_increment_contact_tracing(),
                                               self.instantiate_update_isolation(),
                                               self.instantiate_pcr_testing(),
                                               params)

        # Set the simulated time to the start (days)
        self.time = 0

        # Call parent initialised_simulation
        SimulationModel.simulation_initialised(self)

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

    def instantiate_update_isolation(self) -> UpdateIsolationHousehold:
        return UpdateIsolationHousehold(self.network)

    def instantiate_contact_trace_household(self) -> ContactTraceHousehold:
        return ContactTraceHousehold(self.network)

    def instantiate_increment_contact_tracing(self) -> IncrementContactTracingHousehold:
        return IncrementContactTracingHousehold(self.network)

    def instantiate_pcr_testing(self) -> None:
        return None

    def instantiate_network(self) -> Network:
        return Network()

    def instantiate_new_household(self) -> NewHousehold:
        return NewHousehold(self.network)

    def instantiate_new_infection(self) -> NewInfectionHousehold:
        return NewInfectionHousehold(self.network)

    def instantiate_contact_rate_reduction(self) -> ContactRateReductionHousehold:
        return ContactRateReductionHousehold()

    def perform_recoveries(self):
        """
        Loops over all nodes in the branching process and determine recoveries.

        time - The current time of the process, if a nodes recovery time equals the current time, then it is set to the
        recovered state
        """
        for node in self.network.all_nodes():
            if node.recovery_time == self.time:
                node.recovered = True

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
            #if node.household_id == node.infected_by_node().household_id:
            if node.infected_by_node():
                if (node.infection_status(self.time) == InfectionStatus.unknown_infection) & node.isolated:
                    if node.locally_infected():

                        if self.time >= (node.household().earliest_recognised_symptom_onset(model_time = self.time)
                                         + self.quarantine_duration):
                            node.isolated = False
                            node.completed_isolation = True  
                            node.completed_isolation_reason = 'completed_quarantine'
                            node.completed_isolation_time = self.time
                # For nodes who do not self-report, and are not in the same household as their infector
                # (if they do not self-report they will not isolate; if contact traced, they will be quarantining for
                    # the quarantine duration)
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
                infection_status = node.infection_status(self.time)
                if infection_status in [InfectionStatus.known_infection,
                                        infection_status.self_recognised_infection]:
                    if self.time >= node.symptom_onset_time + self.self_isolation_duration:
                        node.isolated = False
                        node.completed_isolation = True
                        node.completed_isolation_time = self.time
                        node.completed_isolation_reason = 'completed_isolation'

    def simulate_one_step(self):
        """ Simulates one day of the infection and contact tracing.
        """

        # Perform one day of the infection
        self.infection.increment(self.time)
        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        self.isolate_self_reporting_cases()
        # isolate self-reporting-nodes while they wait for tests
        self.contact_tracing.update_isolation(self.time)
        # propagate contact tracing
        for step in range(5):
            self.contact_tracing.increment(self.time)
        # node recoveries
        self.perform_recoveries()
        # release nodes from quarantine or isolation if the time has arrived
        self.release_nodes_from_quarantine_or_isolation()
        # increment time
        self.time += 1


    def run_simulation(self, num_steps: int, infection_threshold: int = 1000) -> None:
        """ Runs the simulation:
                Sets model state,
                Announces start/stopped and step increments to observers

        Arguments:
            num steps -- The number of step increments to perform
            infection_threshold -- The maximum number of infectious nodes allowed, befure stopping stimulation

        Returns:
            None
        """

        # Tell parent simulation started
        SimulationModel.simulation_started(self)

        while type(self.state) is RunningState:
            prev_graph = self.network.graph.copy()

            # This chunk of code executes one step (a days worth of infections and contact tracings)
            self.simulate_one_step()

            # If graph changed, tell parent
            new_graph = self.network.graph
            if not graphs_isomorphic(prev_graph, new_graph):
                SimulationModel.graph_changed(self)

            # Call parent completed step
            SimulationModel.completed_step_increment(self)

            # Simulation ends if num_steps is reached
            if self.time >= num_steps:
                self.state.timed_out()
            elif self.network.count_non_recovered_nodes() == 0:
                self.state.go_extinct()
            elif self.network.count_non_recovered_nodes() > infection_threshold:
                self.state.max_nodes_infectious()

        # Tell parent simulation stopped
        SimulationModel.simulation_stopped(self)


class IndividualLevelContactTracing(HouseholdLevelContactTracing):

    def __init__(self, params: dict):

        # Set default params
        self.number_of_days_to_trace_backwards = 2
        self.number_of_days_to_trace_forwards = 7
        self.recall_probability_fall_off = 1

        # Call superclass constructor (which overwrites defaults with new params if present)
        super().__init__(params)

    @property
    def prob_testing_positive_lfa_func(self) -> Callable[[int], float]:
        return self.contact_tracing.prob_testing_positive_lfa_func

    @prob_testing_positive_lfa_func.setter
    def prob_testing_positive_lfa_func(self, fn: Callable[[int], float]):
        self.contact_tracing.prob_testing_positive_lfa_func = fn

    @property
    def prob_testing_positive_pcr_func(self) -> Callable[[int], float]:
        return self.contact_tracing.prob_testing_positive_pcr_func

    @prob_testing_positive_pcr_func.setter
    def prob_testing_positive_pcr_func(self, fn: Callable[[int], float]):
        self.contact_tracing.prob_testing_positive_pcr_func = fn

    def instantiate_update_isolation(self) -> UpdateIsolationUK:
        return UpdateIsolationUK(self.network)

    def instantiate_contact_trace_household(self) -> ContactTraceHouseholdUK:
        return ContactTraceHouseholdUK(self.network)

    def instantiate_increment_contact_tracing(self) -> IncrementContactTracingUK:
        return IncrementContactTracingUK(self.network,
                                         self.number_of_days_to_trace_backwards,
                                         self.number_of_days_to_trace_forwards,
                                         self.recall_probability_fall_off)

    def instantiate_pcr_testing(self) -> PCRTestingUK:
        return PCRTestingUK(self.network)

class IndividualTracingDailyTesting(IndividualLevelContactTracing):

    def __init__(self, params):

        # Set param defaults

        self.number_of_days_prior_to_LFA_result_to_trace = 2

        self.lateral_flow_testing_duration = 7

        # Call superclass constructor (which overwrites defaults with new params if present)
        super().__init__(params)

    def instantiate_contact_trace_household(self) -> ContactTraceHouseholdContactModelTest:
        return ContactTraceHouseholdContactModelTest(self.network)

    def instantiate_increment_contact_tracing(self) -> IncrementContactTracingContactModelTest:
        return IncrementContactTracingContactModelTest(self.network,
                                                       self.number_of_days_to_trace_backwards,
                                                       self.number_of_days_to_trace_forwards,
                                                       self.recall_probability_fall_off,
                                                       self.number_of_days_prior_to_LFA_result_to_trace)

    def instantiate_pcr_testing(self) -> PCRTestingContactModelTest:
        return PCRTestingContactModelTest(self.network)

    def instantiate_network(self):
        return NetworkContactModel()

    def instantiate_new_household(self) -> NewHouseholdContactModelTest:
        return NewHouseholdContactModelTest(self.network)

    def instantiate_new_infection(self) -> NewInfectionContactModelTest:
        return NewInfectionContactModelTest(self.network)

    def instantiate_contact_rate_reduction(self) -> ContactRateReductionContactModelTest:
        return ContactRateReductionContactModelTest()

    def simulate_one_step(self):
        """ Simulates one day of the infection and contact tracing.
        """

        self.contact_tracing.receive_pcr_test_results(self.time)
        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        self.isolate_self_reporting_cases()
        # isolate self-reporting-nodes while they wait for tests
        self.contact_tracing.update_isolation(self.time)
        # isolate self reporting nodes
        self.contact_tracing.act_on_positive_LFA_tests(self.time)
        # if we require PCR tests, to confirm infection we act on those
        if self.contact_tracing.LFA_testing_requires_confirmatory_PCR:
            self.contact_tracing.act_on_confirmatory_pcr_results(self.time)
        # Perform one day of the infection
        self.infection.increment(self.time)
        # propagate contact tracing
        for _ in range(5):
            self.contact_tracing.increment(self.time)
        # node recoveries
        self.perform_recoveries()
        # release nodes from quarantine or isolation if the time has arrived
        self.release_nodes_from_lateral_flow_testing_or_isolation(self.time)

        # increment time
        self.time += 1

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
                        if self.time >= node.positive_test_time + self.self_isolation_duration:
                            node.isolated = False
                            node.completed_isolation = True
                            node.completed_isolation_time = time
                            node.completed_isolation_reason = 'completed_isolation'
                    else:
                        if self.time >= node.symptom_onset_time + self.self_isolation_duration:
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
