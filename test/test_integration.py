# Household intervention integration tests. Some simple runs of the system with fixed seeds.
import copy
from typing import Tuple, List

import numpy.random
from collections import Counter

import pytest

from household_contact_tracing.network import EdgeType, NodeType, Network
from household_contact_tracing.branching_process_controller import BranchingProcessController
import household_contact_tracing.branching_process_models as bpm
from household_contact_tracing.branching_process_model import BranchingProcessModel


@pytest.fixture
def household_params() -> dict:
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


def count_network(network: Network) -> Tuple[Counter, Counter]:
    """Count the types of nodes and types of edges in the network."""
    node_counts = Counter([node.node_type() for node in network.all_nodes()])
    edge_counts = Counter([edge for edge in network.edge_types()])
    return node_counts, edge_counts


class TestSimpleHousehold:
    """The first implementation of the contact tracing model uses household level contact tracing.
    This means that if a case is detected in a household, all members of the household will
    trace their contacts. When an individual is traced, their entire household goes into
    intervention.
    """

    @staticmethod
    def run_simulation(params: dict, days=10) -> BranchingProcessModel:
        """Run the Household model for 10 days with the given params and return the
        model."""
        controller = BranchingProcessController(bpm.HouseholdLevelTracing(params))
        controller.set_graphic_displays(False)
        controller.run_simulation(days)

        return controller.model

    @staticmethod
    def nodes_isolating_correctly(network: Network) -> List[bool]:
        """Check whether everyone whose household is isolated chooses to isolate."""
        isolating_correctly = []
        for node in network.all_nodes():
            if node.household.isolated:
                if node.infection.isolated:
                    isolating_correctly.append(True)
                else:
                    isolating_correctly.append(False)
            else:
                isolating_correctly.append(True)
        return isolating_correctly

    @staticmethod
    def check_second_level_isolation(network: Network):
        """Check whether all households with a contact tracing index of 2 are isolated."""
        for household in network.all_households:
            if household.contact_tracing_index == 2:
                if household.isolated is False:
                    return False
        return True

    def test_no_isolation_no_reporting(self, household_params):
        """The most basic functionality of the model is to simulate a individual-household
        branching process model of SARS-CoV-2. This includes asymptomatic individuals but
        there is, no symptom reporting or self-intervention.
        Because household transmission is based on intervention, there is no household transmission
        either.
        """
        numpy.random.seed(42)
        network = self.run_simulation(household_params).network
        node_counts, edge_counts = count_network(network)
        # There should be some symptomatic nodes and some asymptomatic but no others.
        assert node_counts[NodeType.symptomatic_will_not_report_infection] > 0
        assert node_counts[NodeType.asymptomatic] > 0
        assert len(node_counts) == 2
        # There is no reporting so there can be no tracing, so all edges have the default type
        assert edge_counts[EdgeType.default] > 0
        assert len(edge_counts) == 1

    def test_reporting_and_isolation(self, household_params):
        """The infection reporting probability is now set to a non-zero value.
        This means that some individuals will develop symptoms, and report them, which initiates
        creation and intervention of the other household members. When a nodes household is isolated
        all the nodes inside are isolated and will not make outside household contacts."""

        # 50% of symptomatic individuals will report their symptoms, and self-isolate
        household_params['infection_reporting_prob'] = 0.5
        household_params['self_isolation_duration'] = 10

        numpy.random.seed(42)
        network = self.run_simulation(household_params).network
        node_counts, edge_counts = count_network(network)
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

    def test_basic_tracing(self, household_params):
        """Contact tracing is now activated. This works at a household level on symptom onset.
        When an infection is discovered in a household, contact tracing attempts are made to all
        connected Households. When a household is reached, only the traced node isolates.
        If a node in a traced household develops symptoms, the whole household is isolated and
        contact tracing is again propagated. Being performed upon symptom onset means that
        testing is not performed."""

        household_params['infection_reporting_prob'] = 0.5
        household_params['self_isolation_duration'] = 10
        household_params['contact_tracing_success_prob'] = 1
        household_params['quarantine_duration'] = 10

        numpy.random.seed(39)
        network = self.run_simulation(household_params).network
        node_counts, edge_counts = count_network(network)
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

        # No intervention should expire by day 10 so all whose household is isolated should
        # be isolating.
        assert all(self.nodes_isolating_correctly(network))

    def test_simple_testing(self, household_params):
        """Simulate an epidemic, with household level contact tracing and testing delays.
        The same contact tracing strategy as before, but a test is required before contact tracing.
        The test is assumed to be 100% accurate, but has a delay associated.
        We don't consider the node or edge types since these should be the same as the previous
        test but the nodes should have a testing delay associated with them.
        """
        household_params['infection_reporting_prob'] = 0.5
        household_params['self_isolation_duration'] = 10
        household_params['contact_tracing_success_prob'] = 1
        household_params['quarantine_duration'] = 10
        household_params['test_before_propagate_tracing'] = True

        numpy.random.seed(42)
        network = self.run_simulation(household_params).network

        assert network.node(1).tracing.testing_delay != 0

        # No intervention should expire by day 10 so all whose household is isolated should
        # be isolating.
        assert all(self.nodes_isolating_correctly(network))

    def test_app_tracing(self, household_params):
        """
        We assign a proportion of the population to have digital contact tracing applications
        installed. If there is a contact tracing attempt between two nodes who both have the app
        installed, then we assume that the contact tracing attempt succeeds with 100% probability,
        and there is no contact tracing delay applied so it is instantaneous.
        """
        household_params['infection_reporting_prob'] = 0.5
        household_params['self_isolation_duration'] = 10
        household_params['contact_tracing_success_prob'] = 1
        household_params['quarantine_duration'] = 10
        household_params['prob_has_trace_app'] = 0.7

        numpy.random.seed(39)
        network = self.run_simulation(household_params).network
        node_counts, edge_counts = count_network(network)
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
        # No intervention should expire by day 10 so all whose household is isolated should
        # be isolating.
        assert all(self.nodes_isolating_correctly(network))

    def test_non_uptake_of_isolation(self, household_params):
        """A percentage of people now refuse to take up intervention when traced."""
        household_params['infection_reporting_prob'] = 0.5
        household_params['self_isolation_duration'] = 10
        household_params['contact_tracing_success_prob'] = 1
        household_params['quarantine_duration'] = 10
        household_params['prob_has_trace_app'] = 0.7
        household_params['node_will_uptake_isolation_prob'] = 0.5

        numpy.random.seed(42)
        network = self.run_simulation(household_params).network
        # Some will choose not to isolate even though their household was instructed to.
        assert not(all(self.nodes_isolating_correctly(network)))
        # People who are isolated still should not infect others.

    def test_imperfect_isolation(self, household_params):
        """We now assume that some nodes do isolate or quarantine, but do it badly. An individual
        doing perfect intervention/quarantine will reduce their outside household contacts by 100%,
        an individual who is imperfectly isolating/quarantining will reduce their contacts by less
        than 100%."""

        household_params['infection_reporting_prob'] = 0.5
        household_params['self_isolation_duration'] = 10
        household_params['contact_tracing_success_prob'] = 1
        household_params['quarantine_duration'] = 10
        household_params['prob_has_trace_app'] = 0.7
        # now, 50% of nodes will isolate, but will do it badly
        household_params['propensity_imperfect_quarantine'] = 0.5
        # a node doing imperfect intervention will reduce their outside household contacts by 75%
        household_params['global_contact_reduction_imperfect_quarantine'] = 0.75

        numpy.random.seed(42)
        model = self.run_simulation(household_params)
        network = model.network
        node_imperfect = [node.tracing_adherence.propensity_imperfect_isolation for node in network.all_nodes()]
        assert any(node_imperfect)
        node_contact_rate_reduction = \
            [model.infection.contact_rate_reduction.get_contact_rate_reduction(node) for node in network.all_nodes()]
        # People who are isolating
        assert 1 in node_contact_rate_reduction
        # People who are imperfectly isolating
        assert 0.75 in node_contact_rate_reduction
        # People who are asymptomatic and just social distancing
        assert 0.3 in node_contact_rate_reduction

    def test_two_step_tracing(self, household_params):
        """In two step tracing people are contact traced if they met an infected person. The
        contacts of the contact traced person are then traced as well."""

        household_params['infection_reporting_prob'] = 0.5
        household_params['contact_tracing_success_prob'] = 1
        household_params['do_2_step'] = False

        numpy.random.seed(42)
        model = self.run_simulation(household_params, 15)
        network = model.network
        hh_idxs = [household.contact_tracing_index > 1 for household in network.all_households]
        # Assert that there are some households with a contact tracing index > 1
        assert any(hh_idxs)
        # All of the second level households should be isolating
        self.check_second_level_isolation(network)


@pytest.fixture()
def individual_params():
    base_params = {
        'outside_household_infectivity_scaling': 0.7,
        'contact_tracing_success_prob': 0.7,
        'asymptomatic_prob': 0.2,
        'asymptomatic_relative_infectivity': 0.35,
        'infection_reporting_prob': 0.5,
        'test_delay': 2,
        'reduce_contacts_by': 0.3,
        'quarantine_duration': 10,
        'number_of_days_to_trace_backwards': 2,
        'number_of_days_to_trace_forwards': 5
    }
    return copy.deepcopy(base_params)


class TestIndividualTracing:
    """Individual level tracing is less thorough model of contract tracing than the household
    model but easier to implement in the real world. With individual level tracing, when an
    individual is infected, their contacts are traced and their household members are quarantined.
    However, contacts of household members are not traced unless they develop symptoms."""

    @staticmethod
    def run_simulation(params: dict, days=10) -> BranchingProcessModel:
        """Run the IndividualTracing model for 10 days with the given params and return the
        model."""
        controller = BranchingProcessController(bpm.IndividualLevelTracing(params))
        controller.set_graphic_displays(False)
        controller.run_simulation(days)

        return controller.model

    def test_simple_individual_model(self, individual_params: dict):
        """Run a basic implementation of the individual level tracing model."""
        numpy.random.seed(42)
        model = self.run_simulation(individual_params)
        network = model.network
        node_counts, edge_counts = count_network(network)
        # Some should be asymptomatic, some should isolating, some should not report infection and
        # some should intend to report but not yet be isolating.
        assert node_counts[NodeType.isolated] > 0
        assert node_counts[NodeType.asymptomatic] > 0
        assert node_counts[NodeType.symptomatic_will_not_report_infection] > 0
        assert node_counts[NodeType.symptomatic_will_report_infection] > 0
        assert len(node_counts) == 4
        # There is reporting and the default is that tracing always succeeds. Infections
        # can be spread within households.
        assert edge_counts[EdgeType.default] > 0
        assert edge_counts[EdgeType.within_house] > 0
        assert edge_counts[EdgeType.between_house] > 0
        assert len(edge_counts) == 3


@pytest.fixture
def daily_testing_params():
    params = {"outside_household_infectivity_scaling": 0.3,
              "contact_tracing_success_prob": 0.7,
              "asymptomatic_prob": 0.2,
              "asymptomatic_relative_infectivity": 0.35,
              "infection_reporting_prob": 0.5,
              "reduce_contacts_by": 0.5,
              "starting_infections": 5,
              "self_isolation_duration": 10,
              "lateral_flow_testing_duration": 14
              }
    return copy.deepcopy(params)


class TestIndividualTracingDailyTesting:
    """The individual level tracing can be extended with daily testing of contacts. Instead of
    traced contacts quarantining, they take daily tests."""

    @staticmethod
    def run_simulation(params: dict, days=10) -> BranchingProcessModel:
        """Run the IndividualTracingDailyTesting model for 10 days with the given params and
        return the model."""
        controller = BranchingProcessController(bpm.IndividualTracingDailyTesting(params))
        controller.set_graphic_displays(False)
        controller.run_simulation(days)

        return controller.model

    @staticmethod
    def prob_positive_pcr(time_relative_to_symptom_onset):
        """This function controls the sensitivity of the pcr test and prevents people testing
        positive as soon as they are infected."""
        if time_relative_to_symptom_onset in [4, 5, 6]:
            return 0.75
        else:
            return 0

    @staticmethod
    def prob_positive_lfa(time_relative_to_symptom_onset):
        """This function controls the sensitivity of the lfa test. A value of 0 is unrealistic,
        but it makes it easier to see nodes being lfa tested since they won't move to
        the intervention status due to lfa testing."""
        return 0

    def test_simple_individual_model(self, daily_testing_params: dict):
        """Run the daily testing with with "no lfa testing only quarantine" policy. This means
        that household contacts are quarantined, and are not lateral flow tested. Only those traced
        via a between household contact tracing are lateral flow tested
        (if they are not already isolating).
        """
        numpy.random.seed(40)
        daily_testing_params["household_positive_policy"] = "only_quarantine"
        model = self.run_simulation(daily_testing_params, 15)
        network = model.network
