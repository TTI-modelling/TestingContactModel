import os
from typing import Callable
from copy import deepcopy

from household_contact_tracing.behaviours.contact_rate_reduction import \
    ContactRateReductionHouseholdLevelTracing, ContactRateReductionIndividualTracingDaily
from household_contact_tracing.behaviours.lft_nodes import lft_nodes
from household_contact_tracing.behaviours.new_household import NewHouseholdLevel, \
    NewHouseholdIndividualTracingDailyTesting
from household_contact_tracing.behaviours.perform_recoveries import perform_recoveries
from household_contact_tracing.behaviours.release_nodes import \
    release_nodes_who_completed_isolation, \
    release_nodes_who_completed_quarantine, release_nodes_who_completed_lateral_flow_testing
from household_contact_tracing.infection import Infection
from household_contact_tracing.network import Network
from household_contact_tracing.simulation_model import BranchingProcessModel
from household_contact_tracing.parameters import validate_parameters
import household_contact_tracing.behaviours.increment_tracing as increment
import household_contact_tracing.behaviours.isolation as isolation

import household_contact_tracing.behaviours.new_infection as new_infection
from household_contact_tracing.simulation_states import RunningState, ExtinctState,\
    MaxNodesInfectiousState, TimedOutState


class HouseholdLevelTracing(BranchingProcessModel):
    """
        A class used to represent a simulation of contact tracing of households only,
         (without contacting every individual and their contacts)


        Attributes
        ----------
        network : Network
            the persistent storage of model data

        infection: Infection
            the processes/behaviours relating to the spread of infection


        Methods
        -------

        run_simulation(self, max_time: int, infection_threshold: int = 1000) -> None
            Runs the simulation up to a maximum number of increments and max allowed number of
            infected nodes.

        simulate_one_step(self)
            Simulates one increment (day) of the infection and contact tracing.

    """
    schema_path = "schemas/household_sim_contact_tracing.json"

    def __init__(self, params: dict):
        """Initializes a household branching process epidemic. Various contact tracing strategies
        can be utilized in an attempt to control the epidemic.

        Args:
            params (dict): A dictionary of parameters that are used in the model.
        """

        self.params = params
        # Parse parameters against schema to check they are valid
        validate_parameters(params, os.path.join(self.root_dir, self.schema_path))
        # Call parent init
        BranchingProcessModel.__init__(self)

        # Set network
        self.network = Network()

        # Set strategies (Strategy pattern)
        self.infection = self._initialise_infection(self.network)

        # Set the simulated time to the start (days)
        self.time = 0

    def _initialise_infection(self, network: Network):
        return Infection(network,
                         NewHouseholdLevel,
                         new_infection.NewInfectionHouseholdLevel,
                         ContactRateReductionHouseholdLevelTracing,
                         self.params)

    def simulate_one_step(self):
        """Simulates one day of the infection and contact tracing."""

        # Perform one day of the infection
        self.infection.increment(self.time)
        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        new_isolation = isolation.HouseholdIsolation(self.network, self.params)
        new_isolation.isolate_self_reporting_cases(self.time)
        # isolate self-reporting-nodes while they wait for tests
        new_isolation.update_households_contact_traced(self.time)
        new_isolation.update_isolation(self.time)
        # propagate contact tracing
        new_increment = increment.IncrementTracingHouseholdLevel(self.network, self.params)
        for step in range(5):
            new_increment.increment_contact_tracing(self.time)
        # node recoveries
        perform_recoveries(self.network, self.time)
        # release nodes from quarantine or isolation if the time has arrived
        release_nodes_who_completed_isolation(self.network, self.time, self.params)
        release_nodes_who_completed_quarantine(self.network, self.time, self.params)
        # increment time
        self.time += 1

    def run_simulation(self, max_time: int, infection_threshold: int = 1000) -> None:
        """ Runs the simulation:
                Sets model state,
                Announces start/stopped and step increments to observers

        Arguments:
            max_time -- The maximum number of step increments to perform (stops if self.time >=
              max_time). Self.time is cumulative throughout multiple calls to run_simulation.
            infection_threshold -- The maximum number of infectious nodes allowed,
              before stopping simulation

        Returns:
            None
        """

        # Switch model to RunningState
        self._state.switch(RunningState, max_time=max_time, infection_threshold=infection_threshold)

        while type(self.state) is RunningState:
            prev_network = deepcopy(self.network)

            # This chunk of code executes one step (a days worth of infections and contact tracings)
            self.simulate_one_step()

            # If graph changed, tell parent
            if not prev_network == self.network:
                BranchingProcessModel.graph_changed(self)

            # Call parent completed step
            super()._completed_step_increment()

            if self.time >= max_time:
                # Simulation ends if max_time is reached
                self.state.switch(TimedOutState,
                                  total_increments=self.time,
                                  non_recovered_nodes=self.network.count_non_recovered_nodes(),
                                  total_nodes=self.network.node_count
                                  )
            elif self.network.count_non_recovered_nodes() == 0:
                # Simulation ends if no more infectious nodes
                self.state.switch(ExtinctState,
                                  total_increments=self.time,
                                  non_recovered_nodes=0,
                                  total_nodes=self.network.node_count)
            elif self.network.count_non_recovered_nodes() > infection_threshold:
                # Simulation ends if number of infectious nodes > threshold
                self.state.switch(MaxNodesInfectiousState,
                                  total_increments=self.time,
                                  non_recovered_nodes=0,
                                  total_nodes=self.network.node_count)

        # Tell parent simulation stopped
        super()._simulation_stopped()


class IndividualLevelTracing(HouseholdLevelTracing):
    """
        A class used to represent a simulation of contact tracing of households along with
        contacting every individual and their contacts, whether they have tested positive or not.
    """
    def __init__(self, params: dict):
        super().__init__(params)
        # Set the test probabilities to default values - may be overridden by user later
        self._prob_lfa_positive = self.default_prob_lfa_positive
        self._prob_pcr_positive = self.default_prob_pcr_positive

    @property
    def prob_lfa_positive(self) -> Callable[[int], float]:
        return self._prob_lfa_positive

    @prob_lfa_positive.setter
    def prob_lfa_positive(self, fn: Callable[[int], float]):
        self._prob_lfa_positive = fn

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

    @staticmethod
    def default_prob_lfa_positive(infectious_age):
        """Default LFA test result probability."""
        if infectious_age in [4, 5, 6]:
            return 1
        else:
            return 0

    schema_path = "schemas/uk_model.json"

    def _initialise_infection(self, network: Network):
        return Infection(network,
                         NewHouseholdLevel,
                         new_infection.NewInfectionHouseholdLevel,
                         ContactRateReductionHouseholdLevelTracing,
                         self.params)

    def simulate_one_step(self):
        """Simulates one day of the infection and contact tracing."""

        # Perform one day of the infection
        self.infection.increment(self.time)
        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        new_isolation = isolation.IndividualIsolation(self.network, self.params)
        new_isolation.isolate_self_reporting_cases(self.time)
        # isolate self-reporting-nodes while they wait for tests
        new_isolation.update_households_contact_traced(self.time)
        new_isolation.update_isolation(self.time)
        # propagate contact tracing
        new_increment = increment.IncrementTracingIndividualLevel(self.network, self.params,
                                                                  self.prob_pcr_positive)
        for step in range(5):
            new_increment.increment_contact_tracing(self.time)
        # node recoveries
        perform_recoveries(self.network, self.time)
        # release nodes from quarantine or isolation if the time has arrived
        release_nodes_who_completed_isolation(self.network, self.time, self.params)
        release_nodes_who_completed_quarantine(self.network, self.time, self.params)
        # increment time
        self.time += 1


class IndividualTracingDailyTesting(IndividualLevelTracing):
    """A class used to represent a simulation of contact tracing of households along with
    contacting every individual and their contacts, whether they have tested positive or not, along
    with daily testing.

        Attributes
        ----------

        Methods
        -------

    """
    schema_path = "schemas/contact_model_test.json"

    def _initialise_infection(self, network: Network):
        return Infection(network,
                         NewHouseholdIndividualTracingDailyTesting,
                         new_infection.NewInfectionIndividualTracingDailyTesting,
                         ContactRateReductionIndividualTracingDaily,
                         self.params)

    def simulate_one_step(self):
        """ Simulates one day of the infection and contact tracing.
        """
        # Set up objects to initialise parameters.
        new_isolation = isolation.DailyTestingIsolation(self.network, self.params)
        new_increment = increment.IncrementTracingIndividualDailyTesting(self.network,
                                                                         self.params,
                                                                         self.prob_pcr_positive)

        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        new_isolation.isolate_self_reporting_cases(self.time)
        # isolate self-reporting-nodes while they wait for tests
        new_isolation.update_households_contact_traced(self.time)
        new_isolation.update_isolation(self.time)
        # isolate self reporting nodes
        positive_nodes = lft_nodes(self.network, self.time, self.prob_lfa_positive, self.params)
        new_isolation.act_on_positive_LFA_tests(self.time, self.prob_pcr_positive, positive_nodes)
        # if we require PCR tests, to confirm infection we act on those
        if new_increment.LFA_testing_requires_confirmatory_PCR:
            new_isolation.act_on_confirmatory_pcr_results(self.time)
        # Perform one day of the infection
        self.infection.increment(self.time)

        # propagate contact tracing
        for _ in range(5):
            new_increment.increment_contact_tracing(self.time)
        # node recoveries
        perform_recoveries(self.network, self.time)
        # release nodes from quarantine or isolation if the time has arrived
        release_nodes_who_completed_isolation(self.network, self.time, self.params)
        release_nodes_who_completed_lateral_flow_testing(self.network, self.time, self.params)

        # increment time
        self.time += 1
