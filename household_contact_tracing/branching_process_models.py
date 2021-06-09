import os
from copy import deepcopy

from household_contact_tracing.behaviours.contact_rate_reduction import \
    ContactRateReductionHouseholdLevelContactTracing, ContactRateReductionIndividualTracingDaily
from household_contact_tracing.behaviours.isolate_self_reporting import isolate_self_reporting_cases
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
from household_contact_tracing.contact_tracing import ContactTracing
import household_contact_tracing.behaviours.isolation as isolation
import household_contact_tracing.behaviours.increment_tracing as increment
import household_contact_tracing.behaviours.new_infection as new_infection
from household_contact_tracing.simulation_states import RunningState, ExtinctState,\
    MaxNodesInfectiousState, TimedOutState


class HouseholdLevelContactTracing(BranchingProcessModel):
    """
        A class used to represent a simulation of contact tracing of households only,
         (without contacting every individual and their contacts)


        Attributes
        ----------
        network : Network
            the persistent storage of model data

        infection: Infection
            the processes/behaviours relating to the spread of infection

        contact_tracing: ContactTracing
            the processes/behaviours relating to the containment of the infection via contact
            tracing


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
        self.contact_tracing = ContactTracing(self.network, self.params)

        # Set the simulated time to the start (days)
        self.time = 0

    def _initialise_infection(self, network: Network):
        return Infection(network,
                         NewHouseholdLevel,
                         new_infection.NewInfectionHouseholdLevel,
                         ContactRateReductionHouseholdLevelContactTracing,
                         self.params)

    def simulate_one_step(self):
        """Simulates one day of the infection and contact tracing."""

        # Perform one day of the infection
        self.infection.increment(self.time)
        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        isolate_self_reporting_cases(self.network, self.time)
        # isolate self-reporting-nodes while they wait for tests
        new_isolation = isolation.UpdateIsolationHouseholdLevel(self.network, self.contact_tracing.household_positive_policy)
        new_isolation.update_isolation(self.time)
        # propagate contact tracing
        new_increment = increment.IncrementTracingHouseholdLevel(self.network, self.contact_tracing.prob_testing_positive_pcr_func, self.contact_tracing.LFA_testing_requires_confirmatory_PCR, self.params)
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


class IndividualLevelContactTracing(HouseholdLevelContactTracing):
    """
        A class used to represent a simulation of contact tracing of households along with
         contacting every individual and their contacts, whether they have tested positive or not.


        Attributes
        ----------
        prob_testing_positive_pcr_func(self) -> Callable[[int], float]
            function that calculates probability of positive PCR test result


        Methods
        -------

    """
    schema_path = "schemas/uk_model.json"

    def _initialise_infection(self, network: Network):
        return Infection(network,
                         NewHouseholdLevel,
                         new_infection.NewInfectionHouseholdLevel,
                         ContactRateReductionHouseholdLevelContactTracing,
                         self.params)

    def simulate_one_step(self):
        """Simulates one day of the infection and contact tracing."""

        # Perform one day of the infection
        self.infection.increment(self.time)
        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        isolate_self_reporting_cases(self.network, self.time)
        # isolate self-reporting-nodes while they wait for tests
        new_isolation = isolation.UpdateIsolationIndividualLevelTracing(self.network, self.contact_tracing.household_positive_policy)
        new_isolation.update_isolation(self.time)
        # propagate contact tracing
        new_increment = increment.IncrementTracingIndividualLevel(self.network,
                                                                  self.contact_tracing.prob_testing_positive_pcr_func,
                                                                  self.contact_tracing.LFA_testing_requires_confirmatory_PCR,
                                                                  self.params)
        for step in range(5):
            new_increment.increment_contact_tracing(self.time)
        # node recoveries
        perform_recoveries(self.network, self.time)
        # release nodes from quarantine or isolation if the time has arrived
        release_nodes_who_completed_isolation(self.network, self.time, self.params)
        release_nodes_who_completed_quarantine(self.network, self.time, self.params)
        # increment time
        self.time += 1


class IndividualTracingDailyTesting(IndividualLevelContactTracing):
    """A class used to represent a simulation of contact tracing of households along with
    contacting every individual and their contacts, whether they have tested positive or not, along
    with daily testing.

        Attributes
        ----------

        Methods
        -------

    """
    schema_path = "schemas/contact_model_test.json"

    def __init__(self, params):

        # Call superclass constructor (which overwrites defaults with new params if present)
        super().__init__(params)

    def _initialise_infection(self, network: Network):
        return Infection(network,
                         NewHouseholdIndividualTracingDailyTesting,
                         new_infection.NewInfectionIndividualTracingDailyTesting,
                         ContactRateReductionIndividualTracingDaily,
                         self.params)

    def simulate_one_step(self):
        """ Simulates one day of the infection and contact tracing.
        """
        # isolate nodes reached by tracing, isolate nodes due to self-reporting
        isolate_self_reporting_cases(self.network, self.time)
        new_isolation = isolation.UpdateIsolationIndividualTracingDailyTesting(self.network,
                                                                               self.contact_tracing.household_positive_policy)
        # isolate self-reporting-nodes while they wait for tests
        new_isolation.update_isolation(self.time)
        # isolate self reporting nodes
        self.contact_tracing.act_on_positive_LFA_tests(self.time)
        # if we require PCR tests, to confirm infection we act on those
        if self.contact_tracing.LFA_testing_requires_confirmatory_PCR:
            self.contact_tracing.act_on_confirmatory_pcr_results(self.time)
        # Perform one day of the infection
        self.infection.increment(self.time)

        # propagate contact tracing
        new_increment = increment.IncrementTracingIndividualDailyTesting(self.network,
                                                                         self.contact_tracing.prob_testing_positive_pcr_func,
                                                                         self.contact_tracing.LFA_testing_requires_confirmatory_PCR,
                                                                         self.params)
        for _ in range(5):
            new_increment.increment_contact_tracing(self.time)
        # node recoveries
        perform_recoveries(self.network, self.time)
        # release nodes from quarantine or isolation if the time has arrived
        release_nodes_who_completed_isolation(self.network, self.time, self.params)
        release_nodes_who_completed_lateral_flow_testing(self.network, self.time, self.params)

        # increment time
        self.time += 1
