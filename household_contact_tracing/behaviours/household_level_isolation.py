from __future__ import annotations

from household_contact_tracing.network import Network


def update_households_contact_traced(network: Network, time: int):
    """Update the contact traced status for all households that have had the
     contact tracing process get there."""
    for household in network.all_households:
        if household.time_until_contact_traced <= time:
            if not household.contact_traced:
                household.update_network()
                household.isolate_if_symptomatic_nodes(time)
                quarantine_traced_node(household)


def update_isolation(network: Network, time: int):
    for node in network.all_nodes():
        if node.time_of_reporting + node.testing_delay == time:
            if not node.household.isolated:
                if not node.household.contact_traced:
                    node.household.isolate_household(time)


def quarantine_traced_node(household):
    traced_node = household.find_traced_node()

    # the traced node should go into quarantine
    if not traced_node.isolated and traced_node.will_uptake_isolation:
        traced_node.isolated = True