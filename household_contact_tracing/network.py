from __future__ import annotations
from typing import Optional, Iterator, List, Tuple, Dict
from enum import Enum

import networkx as nx


class EdgeType(Enum):
    """Describes the source of infection between nodes."""
    default = 0
    within_house = 1
    between_house = 2
    failed_contact_tracing = 3
    app_traced = 4


class NodeType(Enum):
    default = 0
    isolated = 1
    had_contacts_traced = 2
    received_pos_test_pcr = 3
    received_neg_test_pcr = 4
    confirmatory_pos_pcr_test = 5
    confirmatory_neg_pcr_test = 6
    received_pos_test_lfa = 7
    being_lateral_flow_tested_isolated = 8
    being_lateral_flow_tested_not_isolated = 9
    symptomatic_will_report_infection = 10
    symptomatic_will_not_report_infection = 11


class InfectionStatus(Enum):
    known_infection = 0
    self_recognised_infection = 1
    unknown_infection = 2


class TestType(Enum):
    pcr = 0
    lfa = 1


def graphs_isomorphic(graph1: nx.Graph, graph2: nx.Graph) -> bool:
    """Determine whether graphs have identical network structures."""
    return nx.is_isomorphic(graph1, graph2)


class Network:
    def __init__(self):
        self.graph = nx.Graph()
        self.houses: Optional[HouseholdCollection] = None

    @property
    def house_count(self):
        return len(self.houses.house_dict)

    @property
    def node_count(self):
        return nx.number_of_nodes(self.graph)

    @property
    def active_infections(self):
        """Returns a list of nodes who have not yet recovered.

         Nodes can still infect unless they have been isolated.

        Returns:
            list: list of nodes able to infect
        """
        return [node for node in self.all_nodes() if not node.recovered]

    def reset(self):
        # Reset houses
        self.houses = HouseholdCollection(self)

        # Rest the graph of infections
        self.graph = nx.Graph()

    def count_non_recovered_nodes(self) -> int:
        """Returns the number of nodes not in the recovered state."""
        return len([node for node in self.all_nodes() if not node.recovered])

    def get_edge_between_household(self, house1: Household, house2: Household) -> Tuple[int, int]:
        """Get the id's of the two nodes that connect households."""
        for node1 in house1.nodes():
            for node2 in house2.nodes():
                if self.graph.has_edge(node1.node_id, node2.node_id):
                    return node1.node_id, node2.node_id

    def is_edge_app_traced(self, edge: Tuple[int, int]) -> bool:
        """Returns whether two nodes have the contract tracing app."""
        node_1_app = self.node(edge[0]).has_contact_tracing_app
        node_2_app = self.node(edge[1]).has_contact_tracing_app
        return node_1_app and node_2_app

    def add_node(self, node_id, time, generation, household, isolated, will_uptake_isolation,
                 propensity_imperfect_isolation, asymptomatic, symptom_onset_time,
                 pseudo_symptom_onset_time, serial_interval, recovery_time, will_report_infection,
                 time_of_reporting, has_contact_tracing_app, contact_traced, testing_delay=0,
                 additional_attributes: Optional[dict] = None,
                 infecting_node: Optional[Node] = None, completed_isolation=False) -> Node:
        self.graph.add_node(node_id)
        node = Node(nodes=self, houses=self.houses, node_id=node_id, time_infected=time,
                    generation=generation, household=household, isolated=isolated,
                    will_uptake_isolation=will_uptake_isolation,
                    propensity_imperfect_isolation=propensity_imperfect_isolation,
                    asymptomatic=asymptomatic, symptom_onset_time=symptom_onset_time,
                    pseudo_symptom_onset_time=pseudo_symptom_onset_time,
                    serial_interval=serial_interval, recovery_time=recovery_time,
                    will_report_infection=will_report_infection,
                    time_of_reporting=time_of_reporting,
                    has_contact_tracing_app=has_contact_tracing_app,
                    contact_traced=contact_traced,
                    testing_delay=testing_delay,
                    additional_attributes=additional_attributes,
                    infecting_node=infecting_node,
                    completed_isolation=completed_isolation)
        self.graph.nodes[node_id]['node_obj'] = node
        return node

    def node(self, node_id) -> Node:
        return self.graph.nodes[node_id]['node_obj']

    def all_nodes(self) -> Iterator[Node]:
        return (self.node(n) for n in self.graph)

    def asymptomatic_nodes(self) -> Iterator[Node]:
        return [self.node(n) for n in self.graph if self.node(n).asymptomatic]

    def symptomatic_nodes(self) -> Iterator[Node]:
        return [self.node(n) for n in self.graph if not self.node(n).asymptomatic]


class NetworkContractModel(Network):
    def add_node(
        self,
        node_id,
        time,
        generation,
        household,
        isolated,
        will_uptake_isolation,
        propensity_imperfect_isolation,
        asymptomatic,
        symptom_onset_time,
        pseudo_symptom_onset_time,
        serial_interval,
        recovery_time,
        will_report_infection,
        time_of_reporting,
        has_contact_tracing_app,
        contact_traced,
        testing_delay=0,
        additional_attributes: Optional[dict] = None,
        infecting_node: Optional[Node] = None,
        completed_isolation=False,
    ) -> Node:
        self.graph.add_node(node_id)
        node = Node(
            nodes=self,
            houses=self.houses,
            node_id=node_id,
            time_infected=time,
            generation=generation,
            household=household,
            isolated=isolated,
            will_uptake_isolation=will_uptake_isolation,
            propensity_imperfect_isolation=propensity_imperfect_isolation,
            asymptomatic=asymptomatic,
            symptom_onset_time=symptom_onset_time,
            pseudo_symptom_onset_time=pseudo_symptom_onset_time,
            serial_interval=serial_interval,
            recovery_time=recovery_time,
            will_report_infection=will_report_infection,
            time_of_reporting=time_of_reporting,
            has_contact_tracing_app=has_contact_tracing_app,
            testing_delay=testing_delay,
            contact_traced=contact_traced,
            additional_attributes=additional_attributes,
            infecting_node=infecting_node,
            completed_isolation=completed_isolation,
        )
        self.graph.nodes[node_id]['node_obj'] = node
        return node


class Node:
    def __init__(
        self,
        nodes: Network,
        houses: HouseholdCollection,
        node_id: int,
        time_infected: int,
        generation: int,
        household: int,
        isolated: bool,
        will_uptake_isolation: bool,
        propensity_imperfect_isolation: bool,
        asymptomatic: bool,
        symptom_onset_time: float,
        pseudo_symptom_onset_time: int,
        serial_interval: int,
        recovery_time: int,
        will_report_infection: bool,
        time_of_reporting: int,
        has_contact_tracing_app: bool,
        contact_traced: bool,
        testing_delay: int = 0,
        completed_isolation=False,
        had_contacts_traced=False,
        outside_house_contacts_made=0,
        recovered=False,
        infecting_node: Optional[Node] = None,
        additional_attributes: dict = None
    ):
        self.nodes = nodes
        self.houses = houses
        self.node_id = node_id
        self.time_infected = time_infected
        self.generation = generation
        self.household_id = household
        self.isolated = isolated
        self.will_uptake_isolation = will_uptake_isolation
        self.propensity_imperfect_isolation = propensity_imperfect_isolation
        self.asymptomatic = asymptomatic
        self.symptom_onset_time = symptom_onset_time
        self.pseudo_symptom_onset_time = pseudo_symptom_onset_time
        self.serial_interval = serial_interval
        self.recovery_time = recovery_time
        self.will_report_infection = will_report_infection
        self.time_of_reporting = time_of_reporting
        self.has_contact_tracing_app = has_contact_tracing_app
        self.testing_delay = testing_delay
        self.contact_traced = contact_traced
        self.had_contacts_traced = had_contacts_traced
        self.outside_house_contacts_made = outside_house_contacts_made
        self.spread_to_global_node_time_tuples = []
        self.recovered = recovered
        self.time_propagated_tracing = None
        self.propagated_contact_tracing = False
        self.spread_to = []
        self.infected_by_node_id = infecting_node.node_id if infecting_node else None
        self.completed_isolation = completed_isolation
        self.completed_isolation_time = None
        self.completed_isolation_reason = None
        self.completed_traveller_quarantine = False
        self.completed_traveller_lateral_flow_testing = False
        self.received_result = False
        self.received_positive_test_result = None

        self.being_lateral_flow_tested = None
        self.time_started_lfa_testing = None
        self.received_positive_test_result = False
        self.received_result = None
        self.avenue_of_testing: Optional[TestType] = None
        self.positive_test_time = None
        self.node_will_take_up_lfa_testing = None
        self.confirmatory_PCR_result_was_positive: Optional[bool] = None
        self.taken_confirmatory_PCR_test: Optional[bool] = None
        self.confirmatory_PCR_test_time = None
        self.confirmatory_PCR_test_result_time = None
        self.propensity_risky_behaviour_lfa_testing = None
        self.propensity_to_miss_lfa_tests = None

        self.time: Optional[int] = None

        # Update instance variables with anything in params
        for param_name in self.__dict__:
            if param_name in additional_attributes:
                self.__dict__[param_name] = additional_attributes[param_name]

    def household(self) -> Household:
        return self.houses.household(self.household_id)

    def time_relative_to_symptom_onset(self, time: int) -> int:
        # asymptomatics do not have a symptom onset time
        # pseudo_symptom_onset time is a fake onset we give them
        # so we can work out when they test positive
        return time - self.pseudo_symptom_onset_time

    def infected_by_node(self) -> Optional[Node]:
        if self.infected_by_node_id is None:
            return None
        return self.nodes.node(self.infected_by_node_id)

    def locally_infected(self) -> bool:
        if not self.infected_by_node():
            return False
        return self.infected_by_node().household_id == self.household_id

    # TODO: is this the method for deciding if you should isolate (as opposed to quarantine?)
    # MF - I didn't write this method, so I'm unsure
    def has_known_infection(self, time_now: int) -> bool:
        """
        time_now is the model self.time
        Returns:
            bool: Does the node have a known infection
        """
        if self.contact_traced:
            return self.symptom_onset_time >= time_now
        else:
            return self.will_report_infection and self.symptom_onset_time >= time_now

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

    def node_type(self):
        """Returns a node type, given the current status of the node.
        """
        if self.isolated:
            return NodeType.isolated.name
        elif self.had_contacts_traced:
            return NodeType.had_contacts_traced.name
        elif not self.asymptomatic and self.will_report_infection:
            return NodeType.symptomatic_will_report_infection.name
        elif not self.asymptomatic and not self.will_report_infection:
            return NodeType.symptomatic_will_not_report_infection.name
        elif self.received_result and not self.received_positive_test_result and self.avenue_of_testing == TestType.pcr:
            return NodeType.received_neg_test_pcr.name
        elif self.received_positive_test_result and self.avenue_of_testing == TestType.pcr:
            return NodeType.received_pos_test_pcr.name
        elif self.taken_confirmatory_PCR_test and self.confirmatory_PCR_result_was_positive and self.time >= self.confirmatory_PCR_test_result_time:
            return NodeType.confirmatory_pos_pcr_test.name
        elif self.taken_confirmatory_PCR_test and not self.confirmatory_PCR_result_was_positive and self.time >= self.confirmatory_PCR_test_result_time:
            return NodeType.confirmatory_neg_pcr_test.name
        elif self.received_positive_test_result and self.avenue_of_testing == TestType.lfa:
            return NodeType.received_pos_test_lfa.name
        elif self.being_lateral_flow_tested and self.isolated:
            return NodeType.being_lateral_flow_tested_isolated.name
        elif self.being_lateral_flow_tested and not self.isolated:
            return NodeType.being_lateral_flow_tested_not_isolated.name
        else:
            return NodeType.default.name


class Household:

    def __init__(self, houses: HouseholdCollection, nodecollection: Network, house_id: int,
                 house_size: int, time_infected: int, generation: int, infected_by: int,
                 infected_by_node: int, propensity_trace_app: bool,
                 additional_attributes: Optional[dict] = None):
        self.houses = houses
        self.nodecollection = nodecollection
        self.house_id = house_id
        self.size = house_size                  # Size of the household
        self.time_infected = time_infected               # The time at which the infection entered the household
        self.susceptibles = house_size - 1      # How many susceptibles remain in the household
        self.isolated = False                   # Has the household been isolated, so there can be no more infections from this household
        self.isolated_time = float('inf')       # When the house was isolated
        self.propensity_trace_app = propensity_trace_app
        self.contact_traced = False             # If the house has been contact traced, it is isolated as soon as anyone in the house shows symptoms
        self.time_until_contact_traced = float('inf')  # The time until quarantine, calculated from contact tracing processes on connected households
        self.contact_traced_household_ids: List[int] = []  # The list of households contact traced from this one
        self.being_contact_traced_from: Optional[int] = None   # If the house if being contact traced, this is the house_id of the first house that will get there
        self.propagated_contact_tracing = False  # The house has not yet propagated contact tracing
        self.time_propagated_tracing: Optional[int] = None     # Time household propagated contact tracing
        self.contact_tracing_index = 0          # The house is which step of the contact tracing process
        self.generation = generation            # Which generation of households it belongs to
        self.infected_by_id = infected_by       # Which house infected the household
        self.spread_to_ids: List[int] = []          # Which households were infected by this household
        self.node_ids: List[int] = []           # The ID of currently infected nodes in the household
        self.infected_by_node = infected_by_node  # Which node infected the household
        self.within_house_edges: List[Tuple[int, int]] = []  # Which edges are contained within the household
        self.had_contacts_traced = False         # Have the nodes inside the household had their contacts traced?

        self.being_lateral_flow_tested = False,
        self.being_lateral_flow_tested_start_time = None
        self.applied_policy_for_household_contacts_of_a_positive_case = False

        # add custom attributes
        if additional_attributes:
            for key, value in additional_attributes.items():
                setattr(self, key, value)

    def nodes(self) -> Iterator[Node]:
        return (self.nodecollection.node(n) for n in self.node_ids)

    def add_node_id(self, node_id: int):
        self.node_ids.append(node_id)

    def contact_traced_households(self) -> Iterator[Household]:
        return (self.houses.household(hid) for hid in self.contact_traced_household_ids)

    def spread_to(self) -> Iterator[Household]:
        return (self.houses.household(hid) for hid in self.spread_to_ids)

    def infected_by(self) -> Optional[Household]:
        if self.infected_by_id is None:
            return None
        return self.houses.household(self.infected_by_id)

    def has_known_infection(self) -> bool:
        """
        Returns:
            bool: Does the household contain a known infection
        """
        for household_node in self.nodes():
            if max(household_node.symptom_onset_time, self.isolated_time) + household_node.testing_delay >= self.time_infected:
                return True
        return False

    def get_recognised_symptom_onsets(self, model_time: int):
        """Report symptom onset time for all active infections in the Houshold."""
        recognised_symptom_onsets = []

        for household_node in self.nodes():
            infection_status = household_node.infection_status(model_time)
            if infection_status in [InfectionStatus.known_infection,
                                    InfectionStatus.self_recognised_infection]:
                recognised_symptom_onsets.append(household_node.symptom_onset_time)
        return recognised_symptom_onsets

    def get_positive_test_times(self, model_time: int) -> List[int]:
        positive_test_times = []

        for node in self.nodes():
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


class HouseholdCollection:

    def __init__(self, nodes: Network):
        self.house_dict: Dict[int, Household] = {}
        self.nodes = nodes
        # TODO: put house_count in this class

    def add_household(self, house_id: int, house_size: int, time_infected: int, generation: int,
                      infected_by: int, infected_by_node: int, propensity_trace_app: bool,
                      additional_attributes: Optional[dict] = None) -> Household:

        new_household = Household(self, self.nodes, house_id, house_size, time_infected, generation,
                                  infected_by, infected_by_node, propensity_trace_app,
                                  additional_attributes)
        self.house_dict[house_id] = new_household
        return new_household

    def household(self, house_id) -> Household:
        return self.house_dict[house_id]

    @property
    def count(self) -> int:
        return len(self.house_dict)

    def all_households(self) -> Iterator[Household]:
        return (self.household(hid) for hid in self.house_dict)
