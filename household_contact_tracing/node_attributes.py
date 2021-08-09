from typing import Optional, List, Tuple
from household_contact_tracing.parameterised import Parameterised


class InfectionAttributes(Parameterised):
    """
        A class used to store Node attributes relating to infection
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
            asymptomatic
            infecting_node_id
            isolated
            outside_house_contacts_made
            recovered
            recovery_time
            spread_to_global_node_time_tuples
            time_infected
    """

    def __init__(self, attributes: dict):
        self.asymptomatic: Optional[bool] = None
        self.infecting_node_id: Optional[int] = None
        self.isolated: Optional[bool] = None
        self.outside_house_contacts_made: int = 0
        self.recovered: bool = False
        self.recovery_time: Optional[int] = None
        self.spread_to_global_node_time_tuples: List[Tuple[int, int]] = []
        self.time_infected: Optional[int] = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class LFDTestingAttributes(Parameterised):
    """
        A class used to store Node attributes relating to LFD Testing
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
            avenue_of_testing
            being_lateral_flow_tested
            positive_test_time
            taken_confirmatory_PCR_test
            time_started_lfa_testing
            confirmatory_PCR_test_result_time
            completed_lateral_flow_testing_time
            lateral_flow_testing_duration
    """

    def __init__(self, attributes: dict):
        self.avenue_of_testing: Optional[int] = None
        self.being_lateral_flow_tested: Optional[bool] = None
        self.positive_test_time: Optional[int] = None
        self.taken_confirmatory_PCR_test: Optional[bool] = None
        self.time_started_lfa_testing: Optional[int] = None
        self.confirmatory_PCR_test_result_time: Optional[int] = None
        self.completed_lateral_flow_testing_time: Optional[bool] = None
        self.lateral_flow_testing_duration: Optional[int] = 0

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class LFDTestingAdherenceAttributes(Parameterised):
    """
        A class used to store Node attributes relating to LFD Testing Adherence
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
            confirmatory_PCR_result_was_positive
            node_will_take_up_lfa_testing
            propensity_risky_behaviour_lfa_testing
            propensity_to_miss_lfa_tests
    """

    def __init__(self, attributes: dict):
        self.confirmatory_PCR_result_was_positive: Optional[bool] = None
        self.node_will_take_up_lfa_testing: Optional[bool] = None
        self.propensity_risky_behaviour_lfa_testing: Optional[float] = None
        self.propensity_to_miss_lfa_tests: Optional[float] = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class ReturningTravellerAttributes(Parameterised):
    """
        A class used to store Node attributes relating to Returning Travellers
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
            pseudo_symptom_onset_time
    """

    def __init__(self, attributes: dict):
        self.pseudo_symptom_onset_time: Optional[int] = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class TracingAttributes(Parameterised):
    """
        A class used to store Node attributes relating to Contact Tracing
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
            contact_traced
            has_contact_tracing_app
            propagated_contact_tracing
            received_positive_test_result
            received_result
            symptom_onset_time
            testing_delay
            time_of_reporting
            will_report_infection
            completed_isolation
    """

    def __init__(self, attributes: dict):
        self.contact_traced: Optional[bool] = None
        self.has_contact_tracing_app: Optional[bool] = None
        self.propagated_contact_tracing: bool = False
        self.received_positive_test_result: bool = False
        self.received_result: bool = False
        self.symptom_onset_time: Optional[int] = None
        self.testing_delay: Optional[int] = None
        self.time_of_reporting: Optional[int] = None
        self.will_report_infection: Optional[bool] = None
        self.completed_isolation: Optional[bool] = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class TracingAdherenceAttributes(Parameterised):
    """
        A class used to store Node attributes relating to Contact Tracing Adherence
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
        propensity_imperfect_isolation
        will_uptake_isolation
    """

    def __init__(self, attributes: dict):
        self.propensity_imperfect_isolation: Optional[float] = None
        self.will_uptake_isolation: Optional[bool] = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)
