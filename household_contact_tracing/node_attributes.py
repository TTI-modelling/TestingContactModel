from typing import Optional

from household_contact_tracing.parameterised import Parameterised

# Todo: @Martyn Please check


class InfectionAttributes(Parameterised):
    """
        A class used to store Node attributes relating to infection
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
            asymptomatic (boolean)
            infecting_node_id (int)
            isolated (boolean)
            outside_house_contacts_made (int)
            recovered (boolean)
            recovery_time (float)
            spread_to_global_node_time_tuples (list)
            time_infected (int)

        Methods
        -------

    """

    def __init__(self, attributes):
        self.asymptomatic = None
        self.infecting_node_id = None
        self.isolated = None
        self.outside_house_contacts_made = 0
        self.recovered = False
        self.recovery_time = None
        self.spread_to_global_node_time_tuples = []
        self.time_infected = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class LFDTestingAttributes(Parameterised):
    """
        A class used to store Node attributes relating to LFD Testing
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
            avenue_of_testing (int)
            being_lateral_flow_tested (boolean)
            positive_test_time (int)
            taken_confirmatory_PCR_test (boolean)
            time_started_lfa_testing (int)

            # Todo @Martyn:  Ann estimated placing these here - CHECK
            propensity_risky_behaviour_lfa_testing (float)
            propensity_to_miss_lfa_tests (float)
            confirmatory_PCR_test_result_time (float)
            completed_lateral_flow_testing_time (boolean)
            lateral_flow_testing_duration (float)

        Methods
        -------

    """

    def __init__(self, attributes):
        self.avenue_of_testing: Optional[int] = None
        self.being_lateral_flow_tested = None
        self.positive_test_time = None
        self.taken_confirmatory_PCR_test = None
        self.time_started_lfa_testing = None

        # Todo Ann estimated placing these here - CHECK
        self.propensity_risky_behaviour_lfa_testing = None
        self.propensity_to_miss_lfa_tests = None
        self.confirmatory_PCR_test_result_time = None
        self.completed_lateral_flow_testing_time = None
        self.lateral_flow_testing_duration = 0

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class LFDTestingAdherenceAttributes(Parameterised):
    """
        A class used to store Node attributes relating to LFD Testing Adherence
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
            confirmatory_PCR_result_was_positive (boolean)
            node_will_take_up_lfa_testing (boolean)


        Methods
        -------

    """

    def __init__(self, attributes):
        self.confirmatory_PCR_result_was_positive: Optional[bool] = None
        self.node_will_take_up_lfa_testing = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class ReturningTravellerAttributes(Parameterised):
    """
        A class used to store Node attributes relating to Returning Travellers
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
            pseudo_symptom_onset_time (float)

        Methods
        -------
    """

    def __init__(self, attributes):
        self.pseudo_symptom_onset_time = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)


class TracingAttributes(Parameterised):
    """
        A class used to store Node attributes relating to Contact Tracing
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
            contact_traced (boolean)
            has_contact_tracing_app (boolean)
            propagated_contact_tracing (boolean)
            received_positive_test_result (boolean)
            received_result (boolean)
            symptom_onset_time (float)
            testing_delay (float)
            time_of_reporting (int)
            will_report_infection (boolean)
            completed_isolation (boolean)

        Methods
        -------
    """

    def __init__(self, attributes):
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


class TracingAdherenceAttributes(Parameterised):
    """
        A class used to store Node attributes relating to Contact Tracing Adherence
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
        propensity_imperfect_isolation (float)
        will_uptake_isolation (boolean)

        Methods
        -------

    """

    def __init__(self, attributes):
        self.propensity_imperfect_isolation = None
        self.will_uptake_isolation = None

        # Update instance variables with anything in attributes
        self.update_params(attributes)
