from __future__ import annotations

from household_contact_tracing.network import Network, Household


def isolate_self_reporting_cases(network: Network, time: int):
    """Applies the isolation status to nodes who have reached their self-report time.
    They may of course decide to not adhere to said isolation, or may be a member of a household
    who will not uptake isolation
    """
    for node in network.all_nodes():
        if node.will_uptake_isolation:
            if node.time_of_reporting == time:
                node.isolated = True


def update_households_contact_traced(network: Network, time: int):
    """Update the contact traced status for all households that have had the
     contact tracing process get there."""
    for household in network.all_households:
        if household.time_until_contact_traced <= time:
            if not household.contact_traced:
                household.update_network()
                household.isolate_if_symptomatic_nodes(time)
                household.quarantine_traced_node()


def update_isolation(network: Network, time: int):
    for node in network.all_nodes():
        if node.time_of_reporting + node.testing_delay == time:
            if not node.household.isolated:
                if not node.household.contact_traced:
                    node.household.isolate_household(time)
