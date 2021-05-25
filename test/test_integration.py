# Household isolation integration tests. Some simple runs of the system with fixed seeds.


# The first implementation of the contact tracing model uses household level contact tracing.
# By this, we mean that if a case is detected in a household, all members of the household will
# trace their contacts. When an individual is traced, their entire household goes into isolation.


import numpy.random
from collections import Counter

from household_contact_tracing.network import EdgeType, NodeType
from household_contact_tracing.simulation_controller import SimulationController
import household_contact_tracing.branching_process_models as bpm


class TestSimpleHousehold:
    """The most basic functionality of the model is to simulate a individual-household branching
     process model of SARS-CoV-2. This include asymptomatic individuals but there is,
     no symptom reporting or self-isolation.
     """
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
        controller.graph_view.set_display(False)
        controller.graph_pyvis_view.set_display(False)
        controller.timeline_view.set_display(False)
        controller.run_simulation(10)
        network = controller.model.network
        node_counts = Counter([node.node_type() for node in network.all_nodes()])
        edge_counts = Counter([edge for edge in network.edge_types()])
        # There should be some symptomatic nodes and some asymptomatic but no others.
        assert node_counts[NodeType.symptomatic_will_not_report_infection] == 9
        assert node_counts[NodeType.asymptomatic] == 4
        assert len(node_counts) == 2
        # There is no tracing so all edges have the default type
        assert edge_counts[EdgeType.default] == 12
        assert len(edge_counts) == 1
