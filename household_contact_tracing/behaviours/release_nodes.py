from household_contact_tracing.network import InfectionStatus, TestType, Network


def completed_quarantine(network: Network, time: int, params: dict):
    """If a node is currently in quarantine, and has completed the quarantine period then we
    release them from quarantine.

    An individual is in quarantine if they have been contact traced, and have not had symptom onset.

    A quarantined individual is released from quarantine if it has been quarantine_duration since
    they last had contact with a known case.
    In our model, this corresponds to the time of infection.
    """
    if "quarantine_duration" in params:
        quarantine_duration = params["quarantine_duration"]
    else:
        quarantine_duration = 14

    for node in network.all_nodes():
        # For nodes who do not self-report, and are in the same household as their infector
        # (if they do not self-report they will not isolate; if contact traced, they will be
        # quarantining for the quarantine duration)
        # if node.household_id == node.infected_by_node().household_id:
        if node.infecting_node:
            if (node.infection_status(time) == InfectionStatus.unknown_infection) & node.isolated:
                if node.locally_infected():

                    if time >= (node.household.earliest_recognised_symptom_onset(model_time=time)
                                + quarantine_duration):
                        node.isolated = False
                        node.completed_isolation = True
                # For nodes who do not self-report, and are not in the same household as
                # their infector (if they do not self-report they will not isolate; if contact
                # traced, they will be quarantining for the quarantine duration)
                elif node.contact_traced & (time >= node.time_infected + quarantine_duration):
                    node.isolated = False
                    node.completed_isolation = True


def completed_isolation(network: Network, time: int, params: dict):
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
    if "self_isolation_duration" in params:
        self_isolation_duration = params["self_isolation_duration"]
    else:
        self_isolation_duration = 7

    for node in network.all_nodes():
        if node.isolated:
            infection_status = node.infection_status(time)
            if infection_status in [InfectionStatus.known_infection,
                                    InfectionStatus.self_recognised_infection]:
                if node.avenue_of_testing == TestType.lfa:
                    if time >= node.positive_test_time + self_isolation_duration:
                        node.isolated = False
                        node.completed_isolation = True
                else:
                    if time >= node.symptom_onset_time + self_isolation_duration:
                        # this won't include nodes who tested positive due to LF tests who do not
                        # have symptoms
                        node.isolated = False
                        node.completed_isolation = True


def completed_lateral_flow_testing(network: Network, time: int, params: dict):
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
    if "lateral_flow_testing_duration" in params:
        lateral_flow_testing_duration = params["lateral_flow_testing_duration"]
    else:
        lateral_flow_testing_duration = 7

    for node in network.all_nodes():
        if time >= node.time_started_lfa_testing + lateral_flow_testing_duration \
                and node.being_lateral_flow_tested:
            node.being_lateral_flow_tested = False
            node.completed_lateral_flow_testing_time = time
