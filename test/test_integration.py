# Household isolation integration tests. Some simple runs of the system with fixed seeds.
import copy

import numpy.random
from collections import Counter

import pytest

from household_contact_tracing.network import EdgeType, NodeType
from household_contact_tracing.simulation_controller import SimulationController
import household_contact_tracing.branching_process_models as bpm


@pytest.fixture
def params() -> dict:
    """A set of default parameters which are the base for running tests."""
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
              'contact_tracing_success_prob': 0}
    return copy.deepcopy(params)


class TestSimpleHousehold:
    """The first implementation of the contact tracing model uses household level contact tracing.
    This means that if a case is detected in a household, all members of the household will
    trace their contacts. When an individual is traced, their entire household goes into
    isolation.
    """

    def test_no_isolation_no_reporting(self, params):
        """The most basic functionality of the model is to simulate a individual-household
        branching process model of SARS-CoV-2. This includes asymptomatic individuals but
        there is, no symptom reporting or self-isolation.
        Because household transmission is based on isolation, there is no household transmission
        either.
        """
        numpy.random.seed(42)
        controller = SimulationController(bpm.HouseholdLevelContactTracing(params))
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
        # There is no reporting so there can be no tracing, so all edges have the default type
        assert edge_counts[EdgeType.default] == 12
        assert len(edge_counts) == 1

    def test_reporting_and_isolation(self, params):
        """The infection reporting probability is now set to a non-zero value.
        This means that some individuals will develop symptoms, and report them, which initiates
        creation and isolation of the other household members. When a nodes household is isolated
        all the nodes inside are isolated and will not make outside household contacts."""

        # 50% of symptomatic individuals will report their symptoms, and self-isolate
        params['infection_reporting_prob'] = 0.5
        params['self_isolation_duration'] = 10

        numpy.random.seed(42)
        controller = SimulationController(bpm.HouseholdLevelContactTracing(params))
        controller.graph_view.set_display(False)
        controller.graph_pyvis_view.set_display(False)
        controller.timeline_view.set_display(False)
        controller.run_simulation(10)
        network = controller.model.network
        node_counts = Counter([node.node_type() for node in network.all_nodes()])
        edge_counts = Counter([edge for edge in network.edge_types()])
        # Some should be asymptomatic, some should isolate, some should not report infection and
        # some should intend to report but not yet be isolating.
        assert node_counts[NodeType.isolated] == 3
        assert node_counts[NodeType.asymptomatic] == 2
        assert node_counts[NodeType.symptomatic_will_not_report_infection] == 9
        assert node_counts[NodeType.symptomatic_will_report_infection] == 7
        assert len(node_counts) == 4
        # There is reporting but all tracing fails. Now household members are created, infections
        # can be spread within households.
        assert edge_counts[EdgeType.default] == 11
        assert edge_counts[EdgeType.within_house] == 2
        assert edge_counts[EdgeType.failed_contact_tracing] == 7
        assert len(edge_counts) == 3

