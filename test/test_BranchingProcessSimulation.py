import copy

import household_contact_tracing.BranchingProcessSimulation as hct    # The code to test
import pytest

default_params = {"outside_household_infectivity_scaling": 0.8,
                  "contact_tracing_success_prob": 0.7,
                  "overdispersion": 0.32,
                  "asymptomatic_prob": 1,
                  "asymptomatic_relative_infectivity": 0.6,
                  "infection_reporting_prob": 0.5,
                  "household_pairwise_survival_prob": 0.8,
                  "test_delay": 1,
                  "contact_trace_delay": 1,
                  "incubation_period_delay": 5,
                  "symptom_reporting_delay": 1
                  }


def test_asymptomatic_nodes_attributes():

    # everyone's asymptomatic
    test_model = hct.household_sim_contact_tracing(default_params)

    lfa_test_node = test_model.network.node(1)

    assert lfa_test_node.asymptomatic is True
    assert lfa_test_node.symptom_onset_time == float('Inf')
    assert lfa_test_node.will_report_infection is False


def test_symptomatic_nodes_attributes():

    params = copy.deepcopy(default_params)
    params["asymptomatic_prob"] = 0
    params["infection_reporting_prob"] = 1

    # no asymptomatics
    test_model = hct.household_sim_contact_tracing(params)

    lfa_test_node = test_model.network.node(1)

    assert lfa_test_node.asymptomatic is False
    assert lfa_test_node.symptom_onset_time == 5
    assert lfa_test_node.will_report_infection is True


@pytest.fixture
def simple_branching_process():

    params = copy.deepcopy(default_params)

    params["asymptomatic_prob"] = 0.5
    params["asymptomatic_relative_infectivity"] = 0.5

    # 50% asymptomatic
    test_model = hct.household_sim_contact_tracing(params)
    return test_model


def test_global_relative_infectivity(simple_branching_process):

    global_relative_infectivity = simple_branching_process.asymptomatic_global_infection_probs[0] / simple_branching_process.symptomatic_global_infection_probs[0]

    assert global_relative_infectivity == 0.5


def test_get_asymptomatic_infection_prob(simple_branching_process):
    
    local_relative_infectivity = simple_branching_process.asymptomatic_local_infection_probs[1] / simple_branching_process.symptomatic_local_infection_probs[1]

    assert local_relative_infectivity < 0.51
    assert local_relative_infectivity > 0.49
