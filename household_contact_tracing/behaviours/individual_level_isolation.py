from household_contact_tracing.network import Network


def update_households_contact_traced(network: Network, time: int):
    """Update the contact traced status for all households that have had the
     contact tracing process get there."""
    for household in network.all_households:
        if household.time_until_contact_traced <= time:
            if not household.contact_traced:
                household.update_network()
                household.quarantine_traced_node()


def update_isolation(network: Network, time: int):
    for node in network.all_nodes():
        if node.time_of_reporting + node.testing_delay == time:
            if node.received_positive_test_result:
                if not node.household.isolated:
                    if not node.household.contact_traced:
                        node.household.isolate_household(time)