import matplotlib.pyplot as plt

from household_contact_tracing.simulation_controller import SimulationController
from household_contact_tracing.BranchingProcessSimulation import ContactModelTest, uk_model, household_sim_contact_tracing


def prob_testing_positive_function(time_relative_to_symptom_onset):
    if time_relative_to_symptom_onset in [4, 5, 6]:
        return 0.75
    else:
        return 0


def test_delay_dist():
    return 1


def contact_trace_delay_dist():
    return 1


def incubation_period_delay_dist():
    return 5


def symptom_reporting_delay_dist():
    return 1


def main():
    model = ContactModelTest(
        outside_household_infectivity_scaling=0.3,      # How likely an outside contact is to spread
        contact_tracing_success_prob=0.7,               #
        overdispersion=0.32,                            # Variance in number of contacts - Fixed at 0.32
        asymptomatic_prob=0.2,                          #
        asymptomatic_relative_infectivity=0.35,         # Asymptomatic people are less infective
        infection_reporting_prob=0.5,                   # Proportion that report symptoms - how many tracing attempts start
        reduce_contacts_by=0.5,                         # Social distancing parameter - can use to change level of social distancing
        contact_trace=True,
        prob_testing_positive_pcr_func = prob_testing_positive_function,  # Prevents people testing positive as soon as they get it
        prob_testing_positive_lfa_func = prob_testing_positive_function,
        test_delay_dist=test_delay_dist,                # How long people have to wait for a pcr test results (lateral flow is instant)
        contact_trace_delay_dist=contact_trace_delay_dist,  # How long before someone is reached by contact tracing
        incubation_period_delay_dist=incubation_period_delay_dist, # how long between getting it and showing symptoms
        symptom_reporting_delay_dist=symptom_reporting_delay_dist, # how long people wait between getting symptoms and reporting them
        household_pairwise_survival_prob=0.2,               # Probability of infection between household members in household
        lateral_flow_testing_duration=14,                   # How many days people lft for when they are traced
        self_isolation_duration=10,                         # How long people must isolate for when traced
        LFA_testing_requires_confirmatory_PCR=False,
        policy_for_household_contacts_of_a_positive_case='no lfa testing only quarantine'
    )

    controller = SimulationController(model)

    controller.run_simulation(20)

    model.draw_network()
    plt.show()


main()
