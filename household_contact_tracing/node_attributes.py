from typing import Optional, Iterator, List, Tuple, Dict, Callable

from household_contact_tracing.parameterised import Parameterised


class NodeAttributes(Parameterised):
    """

    """
    pass


class InfectionAttributes(NodeAttributes):
    """

    """

    def __init__(self, **attributes):
        self.asymptomatic = None
        self.infecting_node = None
        self.isolated = None
        self.outside_house_contacts_made = 0
        self.recovered = None
        self.recovery_time = None
        self.spread_to_global_node_time_tuples = []
        self.time_infected = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class LFDTestingAttributes(NodeAttributes):
    """

    """

    def __init__(self, **attributes):
        self.avenue_of_testing = None
        self.being_lateral_flow_tested = None
        self.positive_test_time = None
        self.taken_confirmatory_PCR_test = None
        self.time_started_lfa_testing = None

        # Todo Ann estimated location - CHECK
        self.propensity_risky_behaviour_lfa_testing = None
        self.propensity_to_miss_lfa_tests = None
        self.confirmatory_PCR_test_result_time = None
        self.completed_lateral_flow_testing_time = None
        self.lateral_flow_testing_duration = 0

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class LFDTestingAdherenceAttributes(NodeAttributes):
    """

    """

    def __init__(self, **attributes):
        self.confirmatory_PCR_result_was_positive: Optional[bool] = None
        self.node_will_take_up_lfa_testing = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class ReturningTravellerAttributes(NodeAttributes):
    """

    """

    def __init__(self, **attributes):
        self.pseudo_symptom_onset_time = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class TracingAttributes(NodeAttributes):
    """

    """

    def __init__(self, **attributes):
        self.contact_traced = None
        self.has_contact_tracing_app = None
        self.propagated_contact_tracing = False
        self.received_positive_test_result = False
        self.received_result = False
        self.symptom_onset_time = None
        self.testing_delay = None
        self.time_of_reporting = None
        self.will_report_infection = None
        self.completed_isolation = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class TracingAdherenceAttributes(NodeAttributes):
    """

    """

    def __init__(self, **attributes):
        self.propensity_imperfect_isolation = None
        self.will_uptake_isolation = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)
