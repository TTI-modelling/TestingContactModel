from household_contact_tracing.simulation_controller import SimulationController
from household_contact_tracing.BranchingProcessSimulation import ContactModelTest


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
          "policy_for_household_contacts_of_a_positive_case": "no lfa testing only quarantine"}


def main():
    model = ContactModelTest(params)
    model.prob_testing_positive_lfa_func = prob_testing_positive_function
    model.prob_testing_positive_pcr_func = prob_testing_positive_function
    controller = SimulationController(model)
    #controller.set_show_all_graphs(True)
    controller.run_simulation(9)
    controller.run_simulation(11)
    recreate_pytest_1()
    recreate_pytest_2()

def recreate_pytest_1():

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
                      "policy_for_household_contacts_of_a_positive_case": 'lfa testing no quarantine'
                      }

    model = ContactModelTest(default_params)
    model.prob_testing_positive_lfa_func = prob_testing_positive_lfa_func
    model.prob_testing_positive_pcr_func = prob_testing_positive_pcr_func

    model.contact_tracing.contact_tracing_success_prob = 1

    model.infection.new_outside_household_infection(
        time=0,
        infecting_node=model.network.node(1),
        serial_interval=0
    )

    model.contact_tracing.increment_behaviour.attempt_contact_trace_of_household(
        house_to=model.network.houses.household(2),
        house_from=model.network.houses.household(1),
        days_since_contact_occurred=0,
        contact_trace_delay=0,
        time=0
    )

    controller = SimulationController(model=model)
    controller.run_simulation(2)

    print('Assert node 2 being lft\'d', model.network.node(2).being_lateral_flow_tested)

def recreate_pytest_2():
    """Tests that setting the policy option to 'no lfa testing only quarantine' changed the model
    behaviour so if a member of a household tests positive self isolate and start LFA testing.
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
                      "policy_for_household_contacts_of_a_positive_case": 'lfa testing no quarantine'
                      }
    params["policy_for_household_contacts_of_a_positive_case"] = 'lfa testing and quarantine'

    model = ContactModelTest(params)
    model.prob_testing_positive_lfa_func = prob_testing_positive_lfa_func
    model.prob_testing_positive_pcr_func = prob_testing_positive_pcr_func

    model.time = 5

    model.network.node(1).being_lateral_flow_tested = True

    # this line is required before the isolate_positive_lateral_flow_tests func can work
    model.contact_tracing.current_LFA_positive_nodes = model.contact_tracing.get_positive_lateral_flow_nodes(model.time)

    model.contact_tracing.isolate_positive_lateral_flow_tests(model.time)

    model.infection.new_within_household_infection(
        time=model.time,
        infecting_node=model.network.node(1),
        serial_interval=0
    )

    assert model.network.node(1).isolated
    assert model.network.houses.household(1).applied_policy_for_household_contacts_of_a_positive_case
    assert model.network.node(1).received_positive_test_result
    assert model.network.node(2).isolated
    assert model.network.node(2).being_lateral_flow_tested


main()
