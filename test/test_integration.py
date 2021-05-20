# Household isolation integration tests. Some simple runs of the system with fixed seeds.


# The first implementation of the contact tracing model uses household level contact tracing.
# By this, we mean that if a case is detected in a household, all members of the household will
# trace their contacts. When an individual is traced, their entire household goes into isolation.


#The most basic functionality of the model is to simulate a individual-household branching process
# model of SARS-CoV-2. This include asymptomatic individuals but there is, no symptom reporting or
# self-isolation
import numpy.random
from collections import Counter

from household_contact_tracing.simulation_controller import SimulationController
import household_contact_tracing.branching_process_models as bpm


class TestSimpleHousehold:

    params = {'outside_household_infectivity_scaling': 0.7,
              'overdispersion': 0.32,
              'asymptomatic_prob': 0.2,
              'asymptomatic_relative_infectivity': 0.35,
              'infection_reporting_prob': 0,
              'reduce_contacts_by': 0.3,
              'starting_infections': 1,
              'symptom_reporting_delay': 1,
              'incubation_period_delay': 5,
              'household_pairwise_survival_prob': 0.2,
              'contact_trace': False,
              }

    def test_simple_household(self):
        numpy.random.seed(42)
        controller = SimulationController(bpm.HouseholdLevelContactTracing(self.params))
        controller.set_show_graphs(False)
        controller.run_simulation(10)
        network = controller.model.network
        node_counts = Counter([node.node_type().name for node in network.all_nodes()])
        edge_counts = Counter([network.graph.edges[edge]["edge_type"].name for edge in network.graph.edges])
        # There should be some symptomatic nodes and some asyptomatic but no others.
        assert node_counts["symptomatic_will_not_report_infection"] == 9
        assert node_counts["asymptomatic"] == 4
        assert len(node_counts) == 2
        # There is no tracing so all edges have the default type
        assert edge_counts["default"] == 12
        assert len(edge_counts) == 1
