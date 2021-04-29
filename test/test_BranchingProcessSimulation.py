import household_contact_tracing.BranchingProcessSimulation as hct    # The code to test
import pytest

def test_asymptomatic_nodes_attributes():

    def test_delay_dist():
        return 1

    def contact_trace_delay_dist():
        return 1

    def incubation_period_delay_dist():
        return 5

    def symptom_reporting_delay_dist():
        return 1

    # everyone's asymptomatic
    test_model = hct.household_sim_contact_tracing(
        outside_household_infectivity_scaling=0.8,
        contact_tracing_success_prob=0.7,
        overdispersion=0.32,
        asymptomatic_prob=1,
        asymptomatic_relative_infectivity=0.6,
        infection_reporting_prob=0.5,
        contact_trace=True,
        test_delay_dist=test_delay_dist,
        contact_trace_delay_dist=contact_trace_delay_dist,
        incubation_period_delay_dist=incubation_period_delay_dist,
        symptom_reporting_delay_dist=symptom_reporting_delay_dist,
        household_pairwise_survival_prob=0.8
    )

    lfa_test_node = test_model.network.nodes.node(1)

    assert lfa_test_node.asymptomatic == True
    assert lfa_test_node.symptom_onset_time == float('Inf')
    assert lfa_test_node.will_report_infection == False

def test_symptomatic_nodes_attributes():

    def test_delay_dist():
        return 1

    def contact_trace_delay_dist():
        return 1

    def incubation_period_delay_dist():
        return 5

    def symptom_reporting_delay_dist():
        return 1

    # no asymptomatics
    test_model = hct.household_sim_contact_tracing(
        outside_household_infectivity_scaling=0.8,
        contact_tracing_success_prob=0.7,
        overdispersion=0.32,
        asymptomatic_prob=0,
        asymptomatic_relative_infectivity=0.6,
        infection_reporting_prob=1,
        contact_trace=True,
        test_delay_dist=test_delay_dist,
        contact_trace_delay_dist=contact_trace_delay_dist,
        incubation_period_delay_dist=incubation_period_delay_dist,
        symptom_reporting_delay_dist=symptom_reporting_delay_dist,
        household_pairwise_survival_prob=0.8
    )

    lfa_test_node = test_model.network.nodes.node(1)

    assert lfa_test_node.asymptomatic == False
    assert lfa_test_node.symptom_onset_time == 5
    assert lfa_test_node.will_report_infection == True


@pytest.fixture
def simple_branching_process():

    def test_delay_dist():
        return 1

    def contact_trace_delay_dist():
        return 1

    def incubation_period_delay_dist():
        return 5

    def symptom_reporting_delay_dist():
        return 1

    # 50% asymptomatic
    test_model = hct.household_sim_contact_tracing(
        outside_household_infectivity_scaling=0.8,
        contact_tracing_success_prob=0.7,
        overdispersion=0.32,
        asymptomatic_prob=0.5,
        asymptomatic_relative_infectivity=0.5,
        infection_reporting_prob=0.5,
        contact_trace=True,
        test_delay_dist=test_delay_dist,
        contact_trace_delay_dist=contact_trace_delay_dist,
        incubation_period_delay_dist=incubation_period_delay_dist,
        symptom_reporting_delay_dist=symptom_reporting_delay_dist,
        household_pairwise_survival_prob=0.8
    )

    return test_model


def test_global_relative_infectivity(simple_branching_process):

    global_relative_infectivity = simple_branching_process.asymptomatic_global_infection_probs[0] / simple_branching_process.symptomatic_global_infection_probs[0]

    assert global_relative_infectivity == 0.5


def test_get_asymptomatic_infection_prob(simple_branching_process):
    
    local_relative_infectivity = simple_branching_process.asymptomatic_local_infection_probs[1] / simple_branching_process.symptomatic_local_infection_probs[1]

    assert local_relative_infectivity < 0.51
    assert local_relative_infectivity > 0.49
