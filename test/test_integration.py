# Household isolation integration tests. Some simple runs of the system with fixed seeds.
import copy
from typing import Tuple, List

import numpy.random
from collections import Counter

import pytest

from household_contact_tracing.network import EdgeType, NodeType, Network
from household_contact_tracing.simulation_controller import SimulationController
import household_contact_tracing.branching_process_models as bpm
from household_contact_tracing.simulation_model import SimulationModel


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
              'contact_tracing_success_prob': 0,
              'test_before_propagate_tracing': False}
    return copy.deepcopy(params)


class TestSimpleHousehold:
    """The first implementation of the contact tracing model uses household level contact tracing.
    This means that if a case is detected in a household, all members of the household will
    trace their contacts. When an individual is traced, their entire household goes into
    isolation.
    """

    @staticmethod
    def run_simulation(params: dict) -> SimulationModel:
        """Run the Household simulation for 10 steps with the given params and return the
        network."""
        controller = SimulationController(bpm.HouseholdLevelContactTracing(params))
        controller.set_display(False)
        controller.run_simulation(10)

        return controller.model

    @staticmethod
    def count_network(network: Network) -> Tuple[Counter, Counter]:
        """Count the types of nodes and types of edges in the network."""
        node_counts = Counter([node.node_type() for node in network.all_nodes()])
        edge_counts = Counter([edge for edge in network.edge_types()])
        return node_counts, edge_counts

    @staticmethod
    def nodes_isolating_correctly(network: Network) -> List[bool]:
        """Check whether everyone whose household is isolated chooses to isolate."""
        isolating_correctly = []
        for node in network.all_nodes():
            if node.household.isolated:
                if node.isolated:
                    isolating_correctly.append(True)
                else:
                    isolating_correctly.append(False)
            else:
                isolating_correctly.append(True)
        return isolating_correctly

    def test_no_isolation_no_reporting(self, params):
        """The most basic functionality of the model is to simulate a individual-household
        branching process model of SARS-CoV-2. This includes asymptomatic individuals but
        there is, no symptom reporting or self-isolation.
        Because household transmission is based on isolation, there is no household transmission
        either.
        """
        numpy.random.seed(42)
        network = self.run_simulation(params).network
        node_counts, edge_counts = self.count_network(network)
        # There should be some symptomatic nodes and some asymptomatic but no others.
        assert node_counts[NodeType.symptomatic_will_not_report_infection] > 0
        assert node_counts[NodeType.asymptomatic] > 0
        assert len(node_counts) == 2
        # There is no reporting so there can be no tracing, so all edges have the default type
        assert edge_counts[EdgeType.default] > 0
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
        network = self.run_simulation(params).network
        node_counts, edge_counts = self.count_network(network)
        # Some should be asymptomatic, some should isolating, some should not report infection and
        # some should intend to report but not yet be isolating.
        assert node_counts[NodeType.isolated] > 0
        assert node_counts[NodeType.asymptomatic] > 0
        assert node_counts[NodeType.symptomatic_will_not_report_infection] > 0
        assert node_counts[NodeType.symptomatic_will_report_infection] > 0
        assert len(node_counts) == 4
        # There is reporting but all tracing fails. Now household members are created, infections
        # can be spread within households.
        assert edge_counts[EdgeType.default] > 0
        assert edge_counts[EdgeType.within_house] > 0
        assert edge_counts[EdgeType.failed_contact_tracing] > 0
        assert len(edge_counts) == 3

    def test_basic_tracing(self, params):
        """Contact tracing is now activated. This works at a household level on symptom onset.
        When an infection is discovered in a household, contact tracing attempts are made to all
        connected Households. When a household is reached, only the traced node isolates.
        If a node in a traced household develops symptoms, the whole household is isolated and
        contact tracing is again propagated. Being performed upon symptom onset means that
        testing is not performed."""

        params['infection_reporting_prob'] = 0.5
        params['self_isolation_duration'] = 10
        params['contact_tracing_success_prob'] = 1
        params['quarantine_duration'] = 10

        numpy.random.seed(39)
        network = self.run_simulation(params).network
        node_counts, edge_counts = self.count_network(network)
        # As before there are 4 possible node states
        assert node_counts[NodeType.isolated] > 0
        assert node_counts[NodeType.asymptomatic] > 0
        assert node_counts[NodeType.symptomatic_will_not_report_infection] > 0
        assert node_counts[NodeType.symptomatic_will_report_infection] > 0
        assert len(node_counts) == 4
        # The between house edge type is a result of successful contact tracing.
        assert edge_counts[EdgeType.default] > 0
        assert edge_counts[EdgeType.within_house] > 0
        assert edge_counts[EdgeType.between_house] > 0
        assert len(edge_counts) == 3

        # No isolation should expire by day 10 so all whose household is isolated should
        # be isolating.
        assert all(self.nodes_isolating_correctly(network))

    def test_simple_testing(self, params):
        """Simulate an epidemic, with household level contact tracing and testing delays.
        The same contact tracing strategy as before, but a test is required before contact tracing.
        The test is assumed to be 100% accurate, but has a delay associated.
        We don't consider the node or edge types since these should be the same as the previous
        test but the nodes should have a testing delay associated with them.
        """
        params['infection_reporting_prob'] = 0.5
        params['self_isolation_duration'] = 10
        params['contact_tracing_success_prob'] = 1
        params['quarantine_duration'] = 10
        params['test_before_propagate_tracing'] = True

        numpy.random.seed(42)
        network = self.run_simulation(params).network

        assert network.node(1).testing_delay != 0

        # No isolation should expire by day 10 so all whose household is isolated should
        # be isolating.
        assert all(self.nodes_isolating_correctly(network))

    def test_app_tracing(self, params):
        """
        We assign a proportion of the population to have digital contact tracing applications
        installed. If there is a contact tracing attempt between two nodes who both have the app
        installed, then we assume that the contact tracing attempt succeeds with 100% probability,
        and there is no contact tracing delay applied so it is instantaneous.
        """
        params['infection_reporting_prob'] = 0.5
        params['self_isolation_duration'] = 10
        params['contact_tracing_success_prob'] = 1
        params['quarantine_duration'] = 10
        params['prob_has_trace_app'] = 0.7

        numpy.random.seed(39)
        network = self.run_simulation(params).network
        node_counts, edge_counts = self.count_network(network)
        assert node_counts[NodeType.isolated] > 0
        assert node_counts[NodeType.asymptomatic] > 0
        assert node_counts[NodeType.symptomatic_will_not_report_infection] > 0
        assert node_counts[NodeType.symptomatic_will_report_infection] > 0
        assert len(node_counts) == 4
        # The app_traced edge type is a result of app tracing.
        assert edge_counts[EdgeType.default] > 0
        assert edge_counts[EdgeType.within_house] > 0
        assert edge_counts[EdgeType.between_house] > 0
        assert edge_counts[EdgeType.app_traced] > 0
        assert len(edge_counts) == 4
        # No isolation should expire by day 10 so all whose household is isolated should
        # be isolating.
        assert all(self.nodes_isolating_correctly(network))

    def test_non_uptake_of_isolation(self, params):
        """A percentage of people now refuse to take up isolation when traced."""
        params['infection_reporting_prob'] = 0.5
        params['self_isolation_duration'] = 10
        params['contact_tracing_success_prob'] = 1
        params['quarantine_duration'] = 10
        params['prob_has_trace_app'] = 0.7
        params['node_will_uptake_isolation_prob'] = 0.5

        numpy.random.seed(42)
        network = self.run_simulation(params).network
        # Some will choose not to isolate even though their household was instructed to.
        assert not(all(self.nodes_isolating_correctly(network)))
        # People who are isolated still should not infect others.

    def test_imperfect_isolation(self, params):
        """We now assume that some nodes do isolate or quarantine, but do it badly. An individual
        doing perfect isolation/quarantine will reduce their outside household contacts by 100%,
        an individual who is imperfectly isolating/quarantining will reduce their contacts by less
        than 100%."""

        params['infection_reporting_prob'] = 0.5
        params['self_isolation_duration'] = 10
        params['contact_tracing_success_prob'] = 1
        params['quarantine_duration'] = 10
        params['prob_has_trace_app'] = 0.7
        # now, 50% of nodes will isolate, but will do it badly
        params['propensity_imperfect_quarantine'] = 0.5
        # a node doing imperfect isolation will reduce their outside household contacts by 75%
        params['global_contact_reduction_imperfect_quarantine'] = 0.75

        numpy.random.seed(42)
        model = self.run_simulation(params)
        network = model.network
        node_imperfect = [node.propensity_imperfect_isolation for node in network.all_nodes()]
        assert any(node_imperfect)
        node_contact_rate_reduction = [model.infection.contact_rate_reduction_behaviour.get_contact_rate_reduction(node) for node in network.all_nodes()]
        # People who are isolating
        assert 1 in node_contact_rate_reduction
        # People who are imperfectly isolating
        assert 0.75 in node_contact_rate_reduction
        # People who are asymptomatic and just social distancing
        assert 0.3 in node_contact_rate_reduction
