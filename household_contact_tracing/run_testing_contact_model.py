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
    model = ContactModelTest(params, prob_testing_positive_function, prob_testing_positive_function)

    controller = SimulationController(model)

    controller.run_simulation(20)

    controller.set_show_all_graphs(True)
    controller.reset()
    controller.run_simulation(10)


main()
