# Contains objects which encode the network
from typing import Optional, Iterator, List, Tuple, Dict

import networkx as nx


class Node:

    def __init__(
        self,
        nodes: 'NodeCollection',
        houses: 'HouseholdCollection',
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
        testing_delay: int,
        contact_traced: bool,
        completed_isolation=False,
        had_contacts_traced=False,
        outside_house_contacts_made=0,
        recovered=False,
        additional_attributes: Optional[dict] = None,
        infecting_node: Optional['Node']=None,
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

        # add custom attributes
        # This is most useful for model variants
        # first check is for when not supplied, when it will be None
        if additional_attributes:
            for key, value in additional_attributes.items():
                setattr(self, key, value)

    def household(self) -> 'Household':
        return self.houses.household(self.household_id)

    def time_relative_to_symptom_onset(self, time: int) -> int:
        # asymptomatics do not have a symptom onset time
        # pseudo_symptom_onset time is a fake onset we give them
        # so we can work out when they test positive
        return time - self.pseudo_symptom_onset_time

    def infected_by_node(self) -> Optional['Node']:
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


    def infection_status(self, time_now: int) -> str:
        if self.contact_traced:
            if self.symptom_onset_time + self.testing_delay <= time_now:
                return "known_infection"
            if self.symptom_onset_time <= time_now:
                return "self_recognised_infection"
        else:
            if self.will_report_infection:
                if self.time_of_reporting + self.testing_delay <= time_now:
                    return "known_infection"
                if self.time_of_reporting <= time_now:
                    return "self_recognised_infection"
        return "unknown_infection"


class NodeCollection:

    def __init__(self, houses: 'HouseholdCollection'):
        self.G = nx.Graph()
        self.houses = houses

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
        testing_delay,
        contact_traced,
        additional_attributes: Optional[dict] = None,
        infecting_node: Optional[Node]=None,
        completed_isolation=False,
    ) -> Node:
        self.G.add_node(node_id)
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
        self.G.nodes[node_id]['node_obj'] = node
        return node

    def node(self, node_id) -> Node:
        return self.G.nodes[node_id]['node_obj']

    def all_nodes(self) -> Iterator[Node]:
        return (self.node(n) for n in self.G)

    def asymptomatic_nodes(self) -> Iterator[Node]:
        return [self.node(n) for n in self.G if self.node(n).asymptomatic]

    def symptomatic_nodes(self) -> Iterator[Node]:
        return [self.node(n) for n in self.G if not self.node(n).asymptomatic]

class Household:

    def __init__(
        self,
        houses: 'HouseholdCollection',
        nodecollection: NodeCollection,
        house_id: int,
        house_size: int,
        time_infected: int,
        generation: int,
        infected_by: int,
        infected_by_node: int,
        propensity_trace_app: bool,
        additional_attributes: Optional[dict] = None,
    ):
        self.houses = houses
        self.nodecollection = nodecollection
        self.house_id = house_id
        self.size = house_size                  # Size of the household
        self.time = time_infected               # The time at which the infection entered the household
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

        # add custom attributes
        if additional_attributes:
            for key, value in additional_attributes.items():
                setattr(self, key, value)

    def nodes(self) -> Iterator[Node]:
        return (self.nodecollection.node(n) for n in self.node_ids)

    def add_node_id(self, node_id: int):
        self.node_ids.append(node_id)

    def contact_traced_households(self) -> Iterator['Household']:
        return (self.houses.household(hid) for hid in self.contact_traced_household_ids)

    def spread_to(self) -> Iterator['Household']:
        return (self.houses.household(hid) for hid in self.spread_to_ids)

    def infected_by(self) -> 'Household':
        if self.infected_by_id is None:
            return None
        return self.houses.household(self.infected_by_id)

    def has_known_infection(self) -> bool:
        """
        Returns:
            bool: Does the household contain a known infection
        """
        return any([
            True
            for household_node
            in self.nodes()
            # TODO self.time here is household time_infected - consider renaming to self.time_infected
            # to reduce confusion with model self.time
            if max(household_node.symptom_onset_time, self.isolated_time) + household_node.testing_delay >= self.time
        ])

    def earliest_recognised_symptom_onset(self, model_time: int):
        """
        Return infinite if no node in household has recognised symptom onset
        """
        recognised_symptom_onsets = [
            household_node.symptom_onset_time
            for household_node in self.nodes()
            if household_node.infection_status(model_time) in ("known_infection", "self_recognised_infection")
        ]

        if recognised_symptom_onsets != []:
            return min(recognised_symptom_onsets)
        else:
            return float('inf')

    def earliest_recognised_symptom_onset_or_lateral_flow_test(self, model_time: int):
        """
        Return infinite if no node in household has recognised symptom onset
        """
        recognised_symptom_onsets = [
            household_node.symptom_onset_time
            for household_node in self.nodes()
            if household_node.infection_status(model_time) in ("known_infection", "self_recognised_infection")
        ]

        positive_test_times = [
            household_node.positive_test_time
            for household_node in self.nodes()
            if household_node.infection_status(model_time) in ("known_infection")
            and household_node.received_positive_test_result
        ]

        recognised_symptom_and_positive_test_times = recognised_symptom_onsets + positive_test_times

        if recognised_symptom_and_positive_test_times != []:
            return min(recognised_symptom_and_positive_test_times)
        else:
            return float('inf')


class HouseholdCollection:

    def __init__(self, nodes: NodeCollection):
        self.house_dict: Dict[int, Household] = {}
        self.nodes = nodes
        # TODO: put house_count in this class

    def add_household(
        self,
        house_id: int,
        house_size: int,
        time_infected: int,
        generation: int,
        infected_by: int,
        infected_by_node: int,
        propensity_trace_app: bool,
        additional_attributes: Optional[dict] = None) -> Household:
        new_household = Household(
            self,
            self.nodes,
            house_id,
            house_size,
            time_infected,
            generation,
            infected_by,
            infected_by_node,
            propensity_trace_app,
            additional_attributes
        )
        self.house_dict[house_id] = new_household
        return new_household

    def household(self, house_id) -> Household:
        return self.house_dict[house_id]

    @property
    def count(self) -> int:
        return len(self.house_dict)

    def all_households(self) -> Iterator[Household]:
        return (self.household(hid) for hid in self.house_dict)
