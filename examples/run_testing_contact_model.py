from household_contact_tracing.behaviours.intervention.isolation import DailyTestingIsolation
from household_contact_tracing.network import PositivePolicy
from household_contact_tracing.simulation_controller import BranchingProcessController
from household_contact_tracing.branching_process_models import IndividualTracingDailyTesting


def main():
    example_1()
    recreate_pytest_1()
    recreate_pytest_2()


def example_1():

    def prob_testing_positive_function(time_relative_to_symptom_onset):
        # Prevents people testing positive as soon as they get it
        if time_relative_to_symptom_onset in [4, 5, 6]:
            return 0.75
        else:
            return 0

    params = {"outside_household_infectivity_scaling": 0.3,
              "contact_tracing_success_prob": 0.7,
              "overdispersion": 0.32,
              "asymptomatic_prob": 0.2,
              "asymptomatic_relative_infectivity": 0.35,
              "infection_reporting_prob": 0.5,
              "reduce_contacts_by": 0.5,
              "test_delay": 1,
              "contact_trace_delay": 1,
              "incubation_period_delay": 5,
              "symptom_reporting_delay": 1,
              "household_pairwise_survival_prob": 0.2,
              "self_isolation_duration": 10,
              "lateral_flow_testing_duration": 14,
              "LFA_testing_requires_confirmatory_PCR": False,
              "household_positive_policy": PositivePolicy.only_quarantine}

    model = IndividualTracingDailyTesting(params)
    model.prob_lfa_positive = prob_testing_positive_function
    model.prob_pcr_positive = prob_testing_positive_function
    controller = BranchingProcessController(model)
    controller.graph_pyvis_view.set_display(True)
    controller.graph_pyvis_view.open_in_browser = True
    controller.run_simulation(15)
    controller.run_simulation(25)
    controller.graph_pyvis_view.set_display(False)


def recreate_pytest_1():

    def prob_lfa_positive(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 1
        else:
            return 0

    def prob_pcr_positive(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0
        else:
            return 0

    params = {"outside_household_infectivity_scaling": 0.3,
              "contact_tracing_success_prob": 1,
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
              "household_positive_policy": PositivePolicy.lfa_testing_no_quarantine
              }

    model = IndividualTracingDailyTesting(params)
    model.prob_lfa_positive = prob_lfa_positive
    model.prob_pcr_positive = prob_pcr_positive

    model.infection.new_outside_household_infection(time=0, infecting_node=model.network.node(1))
    model.intervention.increment_tracing.prob_pcr_positive = prob_pcr_positive

    model.intervention.increment_tracing.attempt_contact_trace_of_household(
        house_to=model.network.household(2),
        house_from=model.network.household(1),
        days_since_contact_occurred=0,
        contact_trace_delay=0,
        time=0
    )

    controller = BranchingProcessController(model=model)
    controller.shell_view.set_display(False)
    controller.timeline_view.set_display(False)
    controller.graph_pyvis_view.set_display(False)
    controller.run_simulation(2)

    print('Assert node 2 being lft\'d', model.network.node(2).being_lateral_flow_tested)


def recreate_pytest_2():
    """Tests that setting the policy option to 'no lfa testing only quarantine' changed the model
    behaviour so if a member of a household tests positive self isolate and start LFA testing.
    """

    def prob_lfa_positive(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 1
        else:
            return 0

    def prob_pcr_positive(infectious_age):
        if infectious_age in [4, 5, 6]:
            return 0.8
        else:
            return 0

    params = {"outside_household_infectivity_scaling": 0.3,
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
              "household_positive_policy": PositivePolicy.lfa_testing_and_quarantine}

    model = IndividualTracingDailyTesting(params)
    model.prob_lfa_positive = prob_lfa_positive
    model.prob_pcr_positive = prob_pcr_positive

    model.time = 5

    model.network.node(1).being_lateral_flow_tested = True

    positive_nodes = model.intervention.lft_nodes(model.time, prob_lfa_positive)
    new_isolation = DailyTestingIsolation(model.network, params)

    new_isolation.isolate_positive_lateral_flow_tests(model.time, positive_nodes)

    model.infection.new_within_household_infection(time=model.time,
                                                   infecting_node=model.network.node(1))

    assert model.network.node(1).isolated
    assert model.network.household(1).applied_household_positive_policy
    assert model.network.node(1).received_positive_test_result
    assert model.network.node(2).isolated
    assert model.network.node(2).being_lateral_flow_tested


if __name__ == "__main__":
    main()
