import copy
import pytest

from household_contact_tracing.behaviours.intervention.isolation import DailyTestingIsolation
from household_contact_tracing.branching_process_models import IndividualTracingDailyTesting

default_params = {"outside_household_infectivity_scaling": 0.3,
                  "contact_tracing_success_prob": 0.7,
                  "overdispersion": 0.32,
                  "asymptomatic_prob": 0.2,
                  "asymptomatic_relative_infectivity": 0.35,
                  "infection_reporting_prob": 0.3,
                  "LFA_testing_requires_confirmatory_PCR": False,
                  "test_delay": 1,
                  "contact_trace_delay": 1,
                  "incubation_period_delay": 5,
                  "symptom_reporting_delay": 1,
                  "household_pairwise_survival_prob": 0.2,
                  "propensity_risky_behaviour_lfa_testing": 0,
                  "global_contact_reduction_risky_behaviour": 0,
                  "household_positive_policy": "lfa_testing_no_quarantine"
                  }


@pytest.fixture
def simple_model():

    def prob_testing_positive_lfa_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0.5
        else:
            return 0

    def prob_testing_positive_pcr_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0.8
        else:
            return 0

    model = IndividualTracingDailyTesting(default_params)
    model.prob_pcr_positive = prob_testing_positive_pcr_func
    model.prob_lfa_positive = prob_testing_positive_lfa_func

    return model


@pytest.fixture
def simple_model_high_test_prob():
    """The probability of testing positive is 1 on the days either side of symptom onset
    """

    def prob_testing_positive_lfa_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 1
        else:
            return 0

    def prob_testing_positive_pcr_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0
        else:
            return 0

    model = IndividualTracingDailyTesting(default_params)
    model.prob_pcr_positive = prob_testing_positive_pcr_func
    model.prob_lfa_positive = prob_testing_positive_lfa_func

    return model


@pytest.fixture
def simple_model_risky_behaviour():
    """
    Model where no nodes engage in global contacts unless they are being lfa tested 
    and engage in risky behaviour.

    You must manually set the nodes to engage in risky behaviour.
    """

    def prob_testing_positive_lfa_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0.5
        else:
            return 0

    def prob_testing_positive_pcr_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0.8
        else:
            return 0

    params = copy.deepcopy(default_params)
    # all lfa tested nodes engage in risky behaviour
    params["propensity_risky_behaviour_lfa_testing"] = 1

    model = IndividualTracingDailyTesting(params)
    model.prob_pcr_positive = prob_testing_positive_pcr_func
    model.prob_lfa_positive = prob_testing_positive_lfa_func

    return model


def test_pseudo_symptom_onset_asymptomatics():
    """Tests that asymptomatics have a pseudo_symptom_onset_time

    Create a model where every node is asymptomatic

    Check every node has a psuedo symptom onset of 5, see delay below
    """

    def prob_testing_positive_lfa_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 1
        else:
            return 0

    def prob_testing_positive_pcr_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 1
        else:
            return 0

    params = copy.deepcopy(default_params)
    params["asymptomatic_prob"] = 1

    model = IndividualTracingDailyTesting(params)
    model.prob_pcr_positive = prob_testing_positive_pcr_func
    model.prob_lfa_positive = prob_testing_positive_lfa_func

    assert model.network.node(1).returning_travellers.pseudo_symptom_onset_time == 5


def test_pseudo_symptom_onset(simple_model):
    """Checks that it is also working for symptomatics
    """
    assert simple_model.network.node(1).returning_travellers.pseudo_symptom_onset_time == 5


def test_time_relative_to_symptom_onset(simple_model):
    """Tests that nodes return the correct time relative to symptom onset
    """
    # we know symptom onset it at time 5
    assert simple_model.network.node(1).time_relative_to_symptom_onset(5) == 0
    assert simple_model.network.node(1).time_relative_to_symptom_onset(6) == 1
    assert simple_model.network.node(1).time_relative_to_symptom_onset(4) == -1


def test_lfa_test_node(simple_model_high_test_prob):

    model = simple_model_high_test_prob

    node_of_interest = model.network.node(1)

    assert not node_of_interest.lfa_test_node(model.time, model.prob_lfa_positive)

    # set the model time to 5
    # the time relative to symptom onset should now be 0
    # the node should test positive as a result
    model.time = 5

    assert node_of_interest.lfa_test_node(model.time, model.prob_lfa_positive)


def test_being_lateral_flow_tested_attribute(simple_model):
    """Check nodes are generated with the lateral flow testing attribute
    """

    assert not simple_model.network.node(1).lfd_testing.being_lateral_flow_tested


def test_get_positive_lateral_flow_nodes_default_exclusion(simple_model_high_test_prob):
    """Check that nodes are automatically being tested for some reason
    """

    model = simple_model_high_test_prob

    assert model.intervention.lft_nodes(model.time, model.prob_lfa_positive) == []


def test_get_positive_lateral_flow_nodes_timings(simple_model_high_test_prob):
    """Check model is taking timings correctly into account
    """

    model = simple_model_high_test_prob

    node_of_interest = model.network.node(1)

    node_of_interest.lfd_testing.being_lateral_flow_tested = True

    assert model.intervention.lft_nodes(model.time, model.prob_lfa_positive) == []


def test_get_positive_lateral_flow_nodes(simple_model_high_test_prob):
    """Check the node tests positive and is returned by the function
    """

    model = simple_model_high_test_prob

    node_of_interest = model.network.node(1)

    node_of_interest.lfd_testing.being_lateral_flow_tested = True

    model.time = 5

    assert model.intervention.lft_nodes(model.time, model.prob_lfa_positive) == [node_of_interest]


def test_traced_nodes_are_lateral_flow_tested(simple_model_high_test_prob):
    """Checks that a node who is traced is placed under lateral flow testing.

    To do this we:
        * Initialises a model with 100% contact tracing success probability
        * Create a new infection outside the initial household
        * Household 1 traces household 2 with 100% success probability and delay 1
        * Simulate one day twice

    """
    def prob_testing_positive_lfa_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 1
        else:
            return 0

    def prob_testing_positive_pcr_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0
        else:
            return 0

    params = copy.deepcopy(default_params)
    params["contact_tracing_success_prob"] = 1

    model = IndividualTracingDailyTesting(params)
    model.intervention.increment_tracing.prob_pcr_positive = prob_testing_positive_pcr_func
    model.prob_lfa_positive = prob_testing_positive_lfa_func

    model.infection.new_outside_household_infection(time=0, infecting_node=model.network.node(1))

    model.intervention.increment_tracing.attempt_contact_trace_of_household(
        house_to=model.network.household(2),
        house_from=model.network.household(1),
        days_since_contact_occurred=0,
        contact_trace_delay=0,
        time=0
    )

    model.simulate_one_step()
    model.simulate_one_step()

    assert model.network.node(2).lfd_testing.being_lateral_flow_tested is True


def test_isolate_positive_lateral_flow_tests(simple_model_high_test_prob: IndividualTracingDailyTesting):

    model = simple_model_high_test_prob
    model.household_positive_policy = "lfa_testing_and_quarantine"

    model.time = 5

    model.network.node(1).lfd_testing.being_lateral_flow_tested = True

    positive_nodes = model.intervention.lft_nodes(model.time, model.prob_lfa_positive)
    new_isolation = DailyTestingIsolation(model.network, model.params)
    new_isolation.isolate_positive_lateral_flow_tests(model.time, positive_nodes)

    # add another infection to the household, so we can check that they are not quarantining
    # but they are lfa testing
    model.infection.new_within_household_infection(time=model.time, infecting_node=model.network.node(1))

    assert model.network.node(1).infection.isolated
    assert model.network.household(1).applied_household_positive_policy
    assert model.network.node(1).tracing.received_positive_test_result
    assert not model.network.node(2).infection.isolated
    assert model.network.node(2).lfd_testing.being_lateral_flow_tested


@pytest.fixture
def simple_model_lfa_testing_and_quarantine():
    """Individuals who test lfa test are guaranteed to test positive

    Because that makes life easier.

    Used to test the household contact policy option - 'lfa testing and quarantine'
    """

    def prob_testing_positive_lfa_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 1
        else:
            return 0

    def prob_testing_positive_pcr_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0.8
        else:
            return 0

    params = copy.deepcopy(default_params)
    params["household_positive_policy"] = "lfa_testing_and_quarantine"

    model = IndividualTracingDailyTesting(params)
    model.prob_pcr_positive = prob_testing_positive_pcr_func
    model.prob_lfa_positive = prob_testing_positive_lfa_func

    return model


def test_start_lateral_flow_testing_household_and_quarantine(
        simple_model_lfa_testing_and_quarantine: simple_model_lfa_testing_and_quarantine):
    """Tests that setting the policy option to 'lfa testing no quarantine' changed the model
    behaviour so if a member of a household tests positive self isolate and start LFA testing.
    """
    model = simple_model_lfa_testing_and_quarantine
    model.household_positive_policy = "lfa_testing_and_quarantine"

    model.time = 5

    model.network.node(1).lfd_testing.being_lateral_flow_tested = True

    positive_nodes = model.intervention.lft_nodes(model.time, model.prob_lfa_positive)

    isolation = DailyTestingIsolation(model.network, model.params)
    isolation.isolate_positive_lateral_flow_tests(model.time, positive_nodes)

    model.infection.new_within_household_infection(time=model.time,
                                                   infecting_node=model.network.node(1))

    assert model.network.node(1).infection.isolated
    assert model.network.household(1).applied_household_positive_policy
    assert model.network.node(1).tracing.received_positive_test_result
    assert model.network.node(2).infection.isolated
    assert model.network.node(2).lfd_testing.being_lateral_flow_tested


@pytest.fixture
def simple_model_no_lfa_testing_only_quarantine():
    """Individuals who test lfa test are guaranteed to test positive

    Because that makes life easier.

    Used to test the household contact policy option - 'no lfa testing only quarantine'
    """

    def prob_testing_positive_lfa_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 1
        else:
            return 0

    def prob_testing_positive_pcr_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0.8
        else:
            return 0

    params = copy.deepcopy(default_params)

    params["household_positive_policy"] = "only_quarantine"

    model = IndividualTracingDailyTesting(params)
    model.prob_pcr_positive = prob_testing_positive_pcr_func
    model.prob_lfa_positive = prob_testing_positive_lfa_func

    return model


def test_household_contacts_quarantine_only(
    simple_model_lfa_testing_and_quarantine: simple_model_no_lfa_testing_only_quarantine
    ):
    """Tests that setting the policy option to 'no lfa testing only quarantine' changed the model
    behaviour so if a member of a household tests positive self isolate and start LFA testing.
    """
    model = simple_model_lfa_testing_and_quarantine
    model.household_positive_policy = "lfa_testing_and_quarantine"

    model.time = 5

    model.network.node(1).lfd_testing.being_lateral_flow_tested = True

    # this line is required before the isolate_positive_lateral_flow_tests func can work
    positive_nodes = model.intervention.lft_nodes(model.time, model.prob_lfa_positive)
    isolation = DailyTestingIsolation(model.network, model.params)
    isolation.isolate_positive_lateral_flow_tests(model.time, positive_nodes)

    model.infection.new_within_household_infection(time=model.time,
                                                   infecting_node=model.network.node(1))

    assert model.network.node(1).infection.isolated
    assert model.network.household(1).applied_household_positive_policy
    assert model.network.node(1).tracing.received_positive_test_result
    assert model.network.node(2).infection.isolated
    assert model.network.node(2).lfd_testing.being_lateral_flow_tested


def test_risky_behaviour_attributes_default(simple_model: simple_model):
    """Tests that the default behaviour is no more risky behaviour
    """

    assert not simple_model.network.node(1).lfd_testing_adherence.propensity_risky_behaviour_lfa_testing


def test_risky_behaviour_attributes(simple_model_risky_behaviour: simple_model_risky_behaviour):
    """Tests that nodes are created with the propensity for risky behaviour while they are
    being tested.
    """

    assert simple_model_risky_behaviour.network.node(1).lfd_testing_adherence.propensity_risky_behaviour_lfa_testing


@pytest.fixture
def simple_model_risky_behaviour_2_infections():
    """
    Model where no nodes engage in global contacts unless they are being lfa tested 
    and engage in risky behaviour.

    You must manually set the nodes to engage in risky behaviour.
    """

    def prob_testing_positive_lfa_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0.5
        else:
            return 0

    def prob_testing_positive_pcr_func(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0.8
        else:
            return 0

    params = copy.deepcopy(default_params)

    params["starting_infections"] = 2
    params["reduce_contacts_by"] = 1
    params["propensity_risky_behaviour_lfa_testing"] = 1


    model = IndividualTracingDailyTesting(params)
    model.prob_pcr_positive = prob_testing_positive_pcr_func
    model.prob_lfa_positive = prob_testing_positive_lfa_func

    return model


def test_lfa_tested_nodes_make_more_contacts_if_risky(
    simple_model_risky_behaviour_2_infections: simple_model_risky_behaviour_2_infections
    ):
    """Create 2 initial infections, one who engages in risky behaviour while LFA tested 
    and one who doesn't.

    Set both to being lfa tested.

    Sets the household sizes to 1 so that all contact would have to be outside household.

    Assert that the individual who engages in risky behaviour while lfa testing makes contacts
    Assert that the individual does not engage in risky behaviour makes no contacts.
    """ 

    model = simple_model_risky_behaviour_2_infections

    model.network.node(1).lfd_testing_adherence.propensity_risky_behaviour_lfa_testing = False

    # stop there being any within household infections
    # not sure if this is strictly necessary
    model.network.household(1).size = 1
    model.network.household(2).size = 1
    model.network.household(1).susceptibles = 0
    model.network.household(2).susceptibles = 0

    # set the nodes to being lfa tested
    model.network.node(1).lfd_testing.being_lateral_flow_tested = True
    model.network.node(2).lfd_testing.being_lateral_flow_tested = True

    for _ in range(5):
        model.simulate_one_step()

    # node 1 does not engage in risky behaviour and should not make any global contacts
    assert model.network.node(1).infection.outside_house_contacts_made == 0
    assert model.network.node(2).infection.outside_house_contacts_made != 0
