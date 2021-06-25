import sys, os
sys.path.append("../") # REPLACE WITH PATH TO THE 'household_contact_tracing' PACKAGE.

import household_contact_tracing.branching_process_models as bpm
from household_contact_tracing.branching_process_controller import BranchingProcessController


params = {'outside_household_infectivity_scaling': 0.7,
            'contact_tracing_success_prob': 0.0, # doesn't matter, no tracing
            'overdispersion': 0.32,
            'asymptomatic_prob': 0.2,
            'asymptomatic_relative_infectivity': 0.35,
            'infection_reporting_prob': 0,
            'contact_trace': False,
            'test_delay': 2,
            'contact_trace_delay': 1,
            'incubation_period_delay': 5,
            'symptom_reporting_delay': 1,
            'household_pairwise_survival_prob': 0.2,
            'do_2_step': False,
            'reduce_contacts_by': 0.3,
            'prob_has_trace_app': 0,
            'hh_propensity_to_use_trace_app': 1,
            'test_before_propagate_tracing': True,
            'starting_infections': 1,
            'node_will_uptake_isolation_prob': 1,
            'self_isolation_duration': 0,
            'quarantine_duration': 0,
            'transmission_probability_multiplier': 1,
            'propensity_imperfect_quarantine': 0,
            'global_contact_reduction_imperfect_quarantine': 0
         }

# Create controller and add model, then run
controller = BranchingProcessController(bpm.HouseholdLevelTracing(params))
controller.run_simulation(10)

# Update parameters
params['infection_reporting_prob'] = 0.5
params['self_isolation_duration'] = 10

# Re initialise with new parameters and Re-run
controller = BranchingProcessController(bpm.HouseholdLevelTracing(params))
controller.run_simulation(10)

# Add further parameters

params['number_of_days_to_trace_backwards'] = 2
params['number_of_days_to_trace_forwards'] = 5
params['recall_probability_fall_off'] = 1
params['probable_infections_need_test'] = True

# Create new model type
controller = BranchingProcessController(bpm.IndividualLevelTracing(params))
# Switch on a view (e.g. the timeline graph views)
controller.timeline_view.set_display(True)
controller.graph_view.set_display(True)
controller.run_simulation(10)


# Repeat runs and output to a named CSV file (previously defaulted to '/temp/simulation_output_[date-string].csv' )
save_path = os.path.join('..', 'temp', 'my_test.csv')
for idx in range(0, 10):
    controller = BranchingProcessController(bpm.IndividualLevelTracing(params))
    controller.csv_view.filename = save_path
    controller.run_simulation(5)