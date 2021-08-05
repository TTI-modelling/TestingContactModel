from __future__ import annotations
from typing import Optional, Iterator, List, Tuple, Dict, Callable
import networkx as nx
from enum import Enum
import numpy
from dataclasses import dataclass

from household_contact_tracing.parameterised import Parameterised


class EdgeType(Enum):
    """Describes the source of infection between nodes."""
    default = 0
    within_house = 1
    between_house = 2   # This is a contact traced edge
    failed_contact_tracing = 3
    app_traced = 4


class NodeType(Enum):
    """Describes the status of node infection/testing."""
    default = 0
    isolated = 1
    received_pos_test_pcr = 3
    received_neg_test_pcr = 4
    confirmatory_pos_pcr_test = 5
    confirmatory_neg_pcr_test = 6
    received_pos_test_lfa = 7
    being_lateral_flow_tested_isolated = 8
    being_lateral_flow_tested_not_isolated = 9
    symptomatic_will_report_infection = 10
    symptomatic_will_not_report_infection = 11
    asymptomatic = 12


class InfectionStatus(Enum):
    known_infection = 0
    self_recognised_infection = 1
    unknown_infection = 2


class TestType(Enum):
    pcr = 0
    lfa = 1


@dataclass
class EdgeColour:
    colour: str
    label: str


@dataclass
class NodeColour:
    colour: str
    label: str


class Network:
    """
        A class used to store contact tracing data in a graph/network format with nodes and their connecting edges.
        Uses networkx as storage tool.

        Attributes
        ----------
        graph (nx.Graph)
            the persistent storage of graph data

        _house_dict (Dict[int, Household])
            list of households with associated ids

        Methods
        -------
        add_household(self, house_size: int, infected_by: Optional[Household],
                      propensity_trace_app: bool,
                      additional_attributes: Optional[dict] = None) -> Household:

        add_node(self, time_infected, household_id, isolated, will_uptake_isolation,
                 propensity_imperfect_isolation, asymptomatic, symptom_onset_time,
                 pseudo_symptom_onset_time, recovery_time, will_report_infection,
                 time_of_reporting, has_contact_tracing_app, contact_traced, testing_delay=0,
                 additional_attributes: Optional[dict] = None,
                 infecting_node: Optional[Node] = None, completed_isolation=False) -> Node:

    """

    edge_colours = {EdgeType.default: EdgeColour("black", "Transmission, yet to be traced"),
                    EdgeType.within_house: EdgeColour("blue", "Within household contact tracing"),
                    EdgeType.between_house: EdgeColour("magenta",
                                                       "Between household contact tracing"),
                    EdgeType.failed_contact_tracing: EdgeColour("red", "Failed contact trace"),
                    EdgeType.app_traced: EdgeColour("green", "App traced edge")
                    }

    node_colours = {NodeType.default: NodeColour("lightgrey", "Default"),
                    NodeType.isolated: NodeColour('yellow', "Isolating"),
                    NodeType.symptomatic_will_report_infection: NodeColour('lime',
                                                                           "Symptomatic, will report"),
                    NodeType.symptomatic_will_not_report_infection: NodeColour('green',
                                                                               "Symptomatic, will not report"),
                    NodeType.received_pos_test_pcr: NodeColour('grey', "Received positive PCR"),
                    NodeType.received_neg_test_pcr: NodeColour('deeppink', "Received negative PCR"),
                    NodeType.confirmatory_pos_pcr_test: NodeColour('turquoise',
                                                                   "Positive confirmatory PCR"),
                    NodeType.confirmatory_neg_pcr_test: NodeColour('tomato',
                                                                   "Negative confirmatory PCR"),
                    NodeType.received_pos_test_lfa: NodeColour('pink', "Positive LFA"),
                    NodeType.being_lateral_flow_tested_isolated: NodeColour('blue',
                                                                            "Being LFT and isolating"),
                    NodeType.being_lateral_flow_tested_not_isolated: NodeColour('orange',
                                                                                "Being LFT and not isolating"),
                    NodeType.asymptomatic: NodeColour('olive', 'Asymptomatic')
                    }

    def __init__(self):
        self.graph = nx.Graph()
        self._house_dict: Dict[int, Household] = {}

    @property
    def node_count(self):
        return nx.number_of_nodes(self.graph)

    @property
    def house_count(self):
        return len(self._house_dict)

    @property
    def all_households(self) -> Iterator[Household]:
        return (self.household(hid) for hid in self._house_dict)

    @property
    def active_infections(self):
        """Returns a list of nodes who have not yet recovered.

         Nodes can still infect unless they have been isolated.

        Returns:
            list: list of nodes able to infect
        """
        return [node for node in self.all_nodes() if not node.recovered]

    def is_isomorphic(self, network: Network) -> bool:
        """ Determine whether graphs have identical network structures."""
        return nx.is_isomorphic(self.graph, network.graph)

    def __eq__(self, other):
        """ Currently only determines whether graphs have identical network structures,
            but we may want to compare more details.
        """
        return self.is_isomorphic(other)

    def node(self, node_id: int) -> Node:
        """Return the Node from the Network with `node_id`."""
        return self.graph.nodes[node_id]['node_obj']

    def all_nodes(self) -> Iterator[Node]:
        """Return a list of all nodes in the Network"""
        return (self.node(n) for n in self.graph)

    def count_nodes(self, node_type: NodeType) -> int:
        """Returns the number of nodes of type `node_type`."""
        return sum([node.node_type() == node_type for node in self.all_nodes()])

    def add_household(self, house_size: int, infected_by: Optional[Household],
                      propensity_trace_app: bool,
                      additional_attributes: Optional[dict] = None) -> Household:

        new_house_id = self.house_count + 1

        new_household = Household(self, new_house_id, house_size,
                                  infected_by, propensity_trace_app, additional_attributes)
        self._house_dict[new_house_id] = new_household
        return new_household

    def household(self, house_id: int) -> Household:
        return self._house_dict[house_id]

    def count_non_recovered_nodes(self) -> int:
        """Returns the number of nodes not in the recovered state."""
        return len([node for node in self.all_nodes() if not node.recovered])

    def get_edge_between_household(self, house1: Household, house2: Household) -> Tuple[int, int]:
        """Get the id's of the two nodes that connect households."""
        for node1 in house1.nodes:
            for node2 in house2.nodes:
                if self.graph.has_edge(node1.id, node2.id):
                    return node1.id, node2.id

    def is_edge_app_traced(self, edge: Tuple[int, int]) -> bool:
        """Returns whether two nodes have the contract tracing app."""
        node_1_app = self.node(edge[0]).has_contact_tracing_app
        node_2_app = self.node(edge[1]).has_contact_tracing_app
        return node_1_app and node_2_app

    def add_node(self, time_infected, household_id, isolated, will_uptake_isolation,
                 propensity_imperfect_isolation, asymptomatic, symptom_onset_time,
                 pseudo_symptom_onset_time, recovery_time, will_report_infection,
                 time_of_reporting, has_contact_tracing_app, contact_traced, testing_delay=0,
                 additional_attributes: Optional[dict] = None,
                 infecting_node: Optional[Node] = None, completed_isolation=False) -> Node:
        new_node_id = self.node_count + 1
        self.graph.add_node(new_node_id)
        new_node_household = self.household(household_id)
        node = Node(node_id=new_node_id, time_infected=time_infected,
                    household=new_node_household, isolated=isolated,
                    will_uptake_isolation=will_uptake_isolation,
                    propensity_imperfect_isolation=propensity_imperfect_isolation,
                    asymptomatic=asymptomatic, symptom_onset_time=symptom_onset_time,
                    pseudo_symptom_onset_time=pseudo_symptom_onset_time,
                    recovery_time=recovery_time,
                    will_report_infection=will_report_infection,
                    time_of_reporting=time_of_reporting,
                    has_contact_tracing_app=has_contact_tracing_app,
                    contact_traced=contact_traced,
                    testing_delay=testing_delay,
                    additional_attributes=additional_attributes,
                    infecting_node=infecting_node,
                    completed_isolation=completed_isolation)
        self.graph.nodes[new_node_id]['node_obj'] = node
        return node

    def edge_types(self) -> List[EdgeType]:
        """Return a list of edge types in the network."""
        return [self.graph.edges[edge]["edge_type"] for edge in self.graph.edges]

    def label_edges_between_houses(self, house_to: Household, house_from: Household,
                                   new_edge_type: EdgeType):
        """Given two Households, label any edges between the households with `new_edge_type`."""
        for node_1 in house_to.nodes:
            for node_2 in house_from.nodes:
                if self.graph.has_edge(node_1.id, node_2.id):
                    self.graph.edges[node_1.id,
                                     node_2.id].update({"edge_type": new_edge_type})

    def label_edges_inside_household(self, household: Household, new_edge_type: EdgeType):
        """Label all edges within a household with `new_edge_type`."""
        for edge in household.within_house_edges:
            self.graph.edges[edge[0], edge[1]].update({"edge_type": new_edge_type})


class Node(Parameterised):
    """
        A class used to store contact tracing node data.
        Uses networkx as storage tool.
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
        id (int)
            the id of the node

        Methods
        -------
        node_type(self, time=None) -> NodeType:
            Get the NodeType of the node


    """
    def __init__(self, node_id: int, time_infected: int, household: Household, isolated: bool,
                 will_uptake_isolation: bool, propensity_imperfect_isolation: bool,
                 asymptomatic: bool, symptom_onset_time: float, pseudo_symptom_onset_time: int,
                 recovery_time: int, will_report_infection: bool, time_of_reporting: int,
                 has_contact_tracing_app: bool, contact_traced: bool, testing_delay: int = 0,
                 completed_isolation=False, outside_house_contacts_made=0, recovered=False,
                 infecting_node: Optional[Node] = None, additional_attributes: dict = None):

        self.id = node_id
        self.time_infected = time_infected
        self.household = household
        self.isolated = isolated
        self.will_uptake_isolation = will_uptake_isolation
        self.propensity_imperfect_isolation = propensity_imperfect_isolation
        self.asymptomatic = asymptomatic
        self.symptom_onset_time = symptom_onset_time
        self.pseudo_symptom_onset_time = pseudo_symptom_onset_time
        self.recovery_time = recovery_time
        self.will_report_infection = will_report_infection
        self.time_of_reporting = time_of_reporting
        self.has_contact_tracing_app = has_contact_tracing_app
        self.testing_delay = testing_delay
        self.contact_traced = contact_traced
        self.outside_house_contacts_made = outside_house_contacts_made
        self.spread_to_global_node_time_tuples = []
        self.recovered = recovered
        self.propagated_contact_tracing = False
        self.infecting_node = infecting_node if infecting_node else None
        self.completed_isolation = completed_isolation
        self.received_result = False
        self.received_positive_test_result = False

        self.being_lateral_flow_tested = None
        self.time_started_lfa_testing = None
        self.avenue_of_testing: Optional[TestType] = None
        self.positive_test_time = None
        self.node_will_take_up_lfa_testing = None
        self.confirmatory_PCR_result_was_positive: Optional[bool] = None
        self.taken_confirmatory_PCR_test: Optional[bool] = None
        self.confirmatory_PCR_test_result_time = None
        self.propensity_risky_behaviour_lfa_testing = None
        self.propensity_to_miss_lfa_tests = None

        # Update instance variables with anything in `additional_attributes`
        self.update_params(additional_attributes)

    def time_relative_to_symptom_onset(self, time: int) -> int:
        # asymptomatics do not have a symptom onset time
        # pseudo_symptom_onset time is a fake onset we give them
        # so we can work out when they test positive
        return time - self.pseudo_symptom_onset_time

    def locally_infected(self) -> bool:
        if self.infecting_node:
            return self.infecting_node.household == self.household
        else:
            return False

    def infection_status(self, time_now: int) -> InfectionStatus:
        if self.contact_traced:
            if self.symptom_onset_time + self.testing_delay <= time_now:
                return InfectionStatus.known_infection
            if self.symptom_onset_time <= time_now:
                return InfectionStatus.self_recognised_infection
        else:
            if self.will_report_infection:
                if self.time_of_reporting + self.testing_delay <= time_now:
                    return InfectionStatus.known_infection
                if self.time_of_reporting <= time_now:
                    return InfectionStatus.self_recognised_infection
        return InfectionStatus.unknown_infection

    def node_type(self, time=None) -> NodeType:
        """Returns a node type, given the current status of the node.

            params
                time (int): The current increment / step number (e.g. day number) of the simulation
        """
        if self.being_lateral_flow_tested:
            if self.isolated:
                return NodeType.being_lateral_flow_tested_isolated
            else:
                return NodeType.being_lateral_flow_tested_not_isolated
        elif self.isolated:
            return NodeType.isolated
        elif not self.asymptomatic:
            if self.will_report_infection:
                return NodeType.symptomatic_will_report_infection
            else:
                return NodeType.symptomatic_will_not_report_infection
        elif self.received_positive_test_result:
            if self.avenue_of_testing == TestType.pcr:
                return NodeType.received_pos_test_pcr
            else:
                return NodeType.received_pos_test_lfa
        elif self.received_result and self.avenue_of_testing == TestType.pcr:
            return NodeType.received_neg_test_pcr
        elif self.taken_confirmatory_PCR_test:
            # Todo: This may need more correcting, added time=None parameter to get it running.
            #       Was self.time but network/nodes should not need to know about time
            if time and time >= self.confirmatory_PCR_test_result_time:
                if self.confirmatory_PCR_result_was_positive:
                    return NodeType.confirmatory_pos_pcr_test
                else:
                    return NodeType.confirmatory_neg_pcr_test
        elif self.asymptomatic:
            return NodeType.asymptomatic
        else:
            return NodeType.default

    def take_confirmatory_pcr_test(self, time: int, prob_pcr_positive: Callable):
        """Given a the time relative to a nodes symptom onset, will that node test positive."""

        infectious_age_when_tested = time - self.time_infected

        self.confirmatory_PCR_test_result_time = time + self.testing_delay
        self.taken_confirmatory_PCR_test = True

        if numpy.random.binomial(1, prob_pcr_positive(infectious_age_when_tested)) == 1:
            self.confirmatory_PCR_result_was_positive = True

        else:
            self.confirmatory_PCR_result_was_positive = False

    def will_lfa_test_today(self, daily_prob_lfa_test: float) -> bool:
        """Determine whether a node will do an LFT test today."""
        if not self.propensity_to_miss_lfa_tests:
            return True

        if numpy.random.binomial(1, daily_prob_lfa_test) == 1:
            return True
        else:
            return False

    def lfa_test_node(self, time: int, prob_lfa_positive: Callable):
        """Given a the time relative to a nodes symptom onset, will that node test positive"""
        infectious_age = time - self.time_infected

        prob_positive_result = prob_lfa_positive(infectious_age)

        if numpy.random.binomial(1, prob_positive_result) == 1:
            return True
        else:
            return False


class Household:
    """
        A class used to store contact tracing household data.
        Inherits from Node
        Inherits from Parameterised to handle validation and updating of large number of parameters

        Attributes
        ----------
        network (Network)
            The network that this household is a member of
        id (int)
            ID of the household
        size (int)
            The number of nodes contained within the household

        Methods
        -------
        node_type(self, time=None) -> NodeType:
            Get the NodeType of the node
    """
    def __init__(self, network: Network, house_id: int,
                 house_size: int, infected_by: Optional[Household],
                 propensity_trace_app: bool, additional_attributes: Optional[dict] = None):
        self.network = network
        self.id = house_id
        self.size = house_size                  # Size of the household
        self.susceptibles = house_size - 1      # How many susceptibles remain in the household
        # Has the household been isolated, so there can be no more infections from this household
        self.isolated = False
        self.isolated_time = float('inf')       # When the house was isolated
        self.propensity_trace_app = propensity_trace_app
        # If the house has been contact traced, it is isolated as soon as anyone in the house shows symptoms
        self.contact_traced = False
        # The time until quarantine, calculated from contact tracing processes on connected households
        self.time_until_contact_traced = float('inf')
        self.contact_traced_households: List[Household] = []  # The list of households contact traced from this one
        # If the house if being contact traced, this is the first Household that will get there
        self.being_contact_traced_from: Optional[Household] = None
        self.propagated_contact_tracing = False  # The house has not yet propagated contact tracing
        self.contact_tracing_index = 0          # The house is which step of the contact tracing process
        self.infected_by = infected_by       # Which house infected the household
        self.spread_to: List[Household] = []          # Which households were infected by this household
        self.nodes: List[Node] = []           # The ID of currently infected nodes in the household
        self.within_house_edges: List[Tuple[int, int]] = []  # Which edges are contained within the household

        self.being_lateral_flow_tested = False,
        self.being_lateral_flow_tested_start_time = None
        self.applied_household_positive_policy = False

        # add custom attributes
        if additional_attributes:
            for key, value in additional_attributes.items():
                setattr(self, key, value)

    def __eq__(self, other: Household) -> bool:
        if self.id == other.id:
            return True
        else:
            return False

    def get_recognised_symptom_onsets(self, model_time: int):
        """Report symptom onset time for all active infections in the Household."""
        recognised_symptom_onsets = []

        for household_node in self.nodes:
            infection_status = household_node.infection_status(model_time)
            if infection_status in [InfectionStatus.known_infection,
                                    InfectionStatus.self_recognised_infection]:
                recognised_symptom_onsets.append(household_node.symptom_onset_time)
        return recognised_symptom_onsets

    def get_positive_test_times(self, model_time: int) -> List[int]:
        positive_test_times = []

        for node in self.nodes:
            if node.infection_status(model_time) == InfectionStatus.known_infection:
                if node.received_positive_test_result:
                    positive_test_times.append(node.positive_test_time)
        return positive_test_times

    def earliest_recognised_symptom_onset(self, model_time: int):
        """Return infinite if no node in household has recognised symptom onset."""
        recognised_symptom_onsets = self.get_recognised_symptom_onsets(model_time)

        if recognised_symptom_onsets:
            return min(recognised_symptom_onsets)
        else:
            return float('inf')

    def earliest_recognised_symptom_onset_or_lateral_flow_test(self, model_time: int):
        """
        Return infinite if no node in household has recognised symptom onset
        """
        recognised_symptom_onsets = self.get_recognised_symptom_onsets(model_time)
        positive_test_times = self.get_positive_test_times(model_time)

        recognised_symptom_and_positive_test_times = recognised_symptom_onsets + positive_test_times

        if recognised_symptom_and_positive_test_times:
            return min(recognised_symptom_and_positive_test_times)
        else:
            return float('inf')

    def isolate_household(self, time: int):
        """If a Household is contact traced, all Nodes may be required to isolate."""
        if self.isolated:
            return

        # update isolated and contact traced status for Household
        self.contact_traced = True
        self.isolated = True
        self.isolated_time = time

        # Update isolated and contact traced status for Nodes in Household
        for node in self.nodes:
            node.contact_traced = True
            if node.will_uptake_isolation:
                node.isolated = True

        self._update_edges_on_isolation()

    def _update_edges_on_isolation(self):
        """If a house is isolated after being contact traced, update edge colours between
        Households and within the Household."""

        if self.being_contact_traced_from is not None:

            # Initially the edge is assigned the contact tracing label, may be updated if the
            # contact tracing does not succeed
            edge = self.network.get_edge_between_household(self, self.being_contact_traced_from)
            if self.network.is_edge_app_traced(edge):
                self.network.label_edges_between_houses(self, self.being_contact_traced_from,
                                                        EdgeType.app_traced)
            else:
                self.network.label_edges_between_houses(self, self.being_contact_traced_from,
                                                        EdgeType.between_house)

        # Update edges within household
        self.network.label_edges_inside_household(self, EdgeType.within_house)

    def start_lateral_flow_testing_household(self, time: int):
        """Sets the household to the lateral flow testing status so that new within household
        infections are tested. All nodes in the household start lateral flow testing
        """
        self.being_lateral_flow_tested = True
        self.being_lateral_flow_tested_start_time = time

        for node in self.nodes:
            if node.node_will_take_up_lfa_testing:
                if not node.received_positive_test_result:
                    if not node.being_lateral_flow_tested:
                        node.being_lateral_flow_tested = True
                        node.time_started_lfa_testing = time

    def start_lateral_flow_testing_household_and_quarantine(self, time):
        """Sets the household to the lateral flow testing status so that new within household
        infections are tested. All nodes in the household start lateral flow testing and
        start quarantining
        """
        self.being_lateral_flow_tested = True
        self.being_lateral_flow_tested_start_time = time
        self.isolated = True
        self.isolated_time = True
        self.contact_traced = True

        for node in self.nodes:
            if node.node_will_take_up_lfa_testing:
                if not node.received_positive_test_result:
                    if not node.being_lateral_flow_tested:
                        node.being_lateral_flow_tested = True
                        node.time_started_lfa_testing = time

            if node.will_uptake_isolation:
                node.isolated = True

    def apply_positive_policy(self, time: int, household_positive_policy: str):
        """Depending on the positive policy, different interventions are made to the household
        contacts of a discovered case.

        lfa_testing_no_quarantine: Household contacts start LFA testing, but do not quarantine
          unless they develop symptoms.
        lfa_testing_and_quarantine: Household contacts start LFA testing and quarantine.
        only_quarantine: Household contacts do not start LFA testing, quarantine. They will book
          a PCR test if they develop symptoms.
        """

        # set the household attributes to declare that we have already applied the policy
        self.applied_household_positive_policy = True

        if household_positive_policy == "lfa_testing_no_quarantine":
            self.start_lateral_flow_testing_household(time)
        elif household_positive_policy == "lfa_testing_and_quarantine":
            self.start_lateral_flow_testing_household_and_quarantine(time)
        elif household_positive_policy == "only_quarantine":
            self.isolate_household(time)
        else:
            raise Exception("household_positive_policy not recognised.")

    def find_traced_node(self):
        """Work out which was the traced node."""
        tracing_household = self.being_contact_traced_from
        traced_node_id = self.network.get_edge_between_household(self, tracing_household)[0]
        return self.network.node(traced_node_id)

    def update_network(self):
        """
        When a house is contact traced, we need to place all the nodes under surveillance.
        If any of the nodes are symptomatic, we need to isolate the household.
        """
        # Update the house to the contact traced status
        self.contact_traced = True

        # Update the nodes to the contact traced status
        for node in self.nodes:
            node.contact_traced = True

        # Colour the edges within household
        self.network.label_edges_inside_household(self, EdgeType.within_house)

    def isolate_if_symptomatic_nodes(self, time: int):
        """If there are any symptomatic nodes in the household then isolate the household."""
        for node in self.nodes:
            if node.symptom_onset_time <= time and not node.completed_isolation:
                self.isolate_household(time)
                break

    def quarantine_traced_node(self):
        traced_node = self.find_traced_node()

        # the traced node should go into quarantine
        if not traced_node.isolated and traced_node.will_uptake_isolation:
            traced_node.isolated = True

    @property
    def local_epidemic_completed(self):
        """
        Returns true if all infections in the household have recovered,
        which is defined as being 10
        """
        return all([node.recovered for node in self.nodes])

    @property
    def household_epidemic_size(self):
        """Returns the current size of the household epidemic, i.e: the number of household members
        that are, or were, infected.
        """
        return self.size - self.susceptibles
