from __future__ import annotations
from typing import Callable, List, Type

from household_contact_tracing.network import Network, Node, InfectionStatus, TestType
from household_contact_tracing.parameterised import Parameterised
from household_contact_tracing.behaviours.isolation import Isolation
from household_contact_tracing.behaviours.increment_tracing import IncrementTracing


class Intervention(Parameterised):
    """
        Logic for performing interventions and daily increment of testing and tracing.

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

    def __init__(self,
                 network: Network,
                 isolation: Type[Isolation],
                 increment_tracing: Type[IncrementTracing],
                 params: dict):

        # Set the network
        self.network = network

        # set intervention input parameters (create defaults then override with inputs if present)
        self.node_daily_prob_lfa_test = 1
        self.quarantine_duration = 14
        self.self_isolation_duration = 7
        self.lateral_flow_testing_duration = 7
        self.update_params(params)

        # Set intervention behaviours
        self.isolation = isolation(network, params)
        self.increment_tracing = increment_tracing(network, params)

    def lft_nodes(self, time: int, prob_lfa_positive: Callable) -> List[Node]:
        """Performs a days worth of lateral flow testing.

        Returns:
            A list of nodes who have tested positive through the lateral flow tests.
        """

        positive_nodes = []
        for node in self.network.all_nodes():
            if node.being_lateral_flow_tested:
                if node.will_lfa_test_today(self.node_daily_prob_lfa_test):
                    if not node.received_positive_test_result:
                        if node.lfa_test_node(time, prob_lfa_positive):
                            positive_nodes.append(node)
        return positive_nodes

    def completed_quarantine(self, time: int):
        """If a node is currently in quarantine, and has completed the quarantine period then we
        release them from quarantine.

        An individual is in quarantine if they have been contact traced, and have not had symptom onset.

        A quarantined individual is released from quarantine if it has been quarantine_duration since
        they last had contact with a known case.
        In our model, this corresponds to the time of infection.
        """

        for node in self.network.all_nodes():
            # For nodes who do not self-report, and are in the same household as their infector
            # (if they do not self-report they will not isolate; if contact traced, they will be
            # quarantining for the quarantine duration)
            # if node.household_id == node.infected_by_node().household_id:
            if node.infecting_node:
                if (node.infection_status(time) == InfectionStatus.unknown_infection) & node.isolated:
                    if node.locally_infected():

                        if time >= (node.household.earliest_recognised_symptom_onset(model_time=time)
                                    + self.quarantine_duration):
                            node.isolated = False
                            node.completed_isolation = True
                    # For nodes who do not self-report, and are not in the same household as
                    # their infector (if they do not self-report they will not isolate; if contact
                    # traced, they will be quarantining for the quarantine duration)
                    elif node.contact_traced & (time >= node.time_infected + self.quarantine_duration):
                        node.isolated = False
                        node.completed_isolation = True

    def completed_isolation(self, time: int):
        """
        Nodes leave self-isolation, rather than quarantine, when their infection status is either known
        (ie tested) or when they are in a contact traced household and they develop symptoms (they
        might then go on to get a test, but they isolate regardless). Nodes in contact traced
        households do not have a will_report_infection probability: if they develop symptoms, they
        are a self-recognised infection who might or might not go on to test and become a known
        infection.

        If it has been isolation_duration since these individuals have had symptom onset, then they are
        released from isolation.
        """

        for node in self.network.all_nodes():
            if node.isolated:
                infection_status = node.infection_status(time)
                if infection_status in [InfectionStatus.known_infection,
                                        InfectionStatus.self_recognised_infection]:
                    if node.avenue_of_testing == TestType.lfa:
                        if time >= node.positive_test_time + self.self_isolation_duration:
                            node.isolated = False
                            node.completed_isolation = True
                    else:
                        if time >= node.symptom_onset_time + self.self_isolation_duration:
                            # this won't include nodes who tested positive due to LF tests who do not
                            # have symptoms
                            node.isolated = False
                            node.completed_isolation = True

    def completed_lateral_flow_testing(self, time: int):
        """If a node is currently in lateral flow testing, and has completed this period then we
        release them from testing.

        An individual is in lateral flow testing if they have been contact traced, and have not had
        symptom onset.

        They continue to be lateral flow tested until the duration of this period is up OR they
        test positive on lateral flow and they are isolated and traced.

        A lateral flow tested individual is released from testing if it has been
        'lateral_flow_testing_duration' since they last had contact with a known case.
        In our model, this corresponds to the time of infection.
        """

        for node in self.network.all_nodes():
            if time >= node.time_started_lfa_testing + self.lateral_flow_testing_duration \
                    and node.being_lateral_flow_tested:
                node.being_lateral_flow_tested = False
                node.completed_lateral_flow_testing_time = time
