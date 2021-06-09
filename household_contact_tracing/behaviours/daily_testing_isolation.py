from household_contact_tracing.network import Network, PositivePolicy, TestType


def update_households_contact_traced(network: Network, time: int):
    """Update the contact traced status for all households that have had the
     contact tracing process get there."""
    for household in network.all_households:
        if household.time_until_contact_traced <= time:
            if not household.contact_traced:
                household.update_network()
                traced_node = household.find_traced_node()
                # the traced node is now being lateral flow tested
                if traced_node.node_will_take_up_lfa_testing:
                    if not traced_node.received_positive_test_result:
                        traced_node.being_lateral_flow_tested = True
                        traced_node.time_started_lfa_testing = time


def update_isolation(network: Network, time: int, household_positive_policy: PositivePolicy):
    for node in network.all_nodes():
        if node.positive_test_time == time:
            if node.avenue_of_testing == TestType.pcr:
                if node.received_positive_test_result:
                    if not node.household.applied_household_positive_policy:
                        node.household.apply_positive_policy(time, household_positive_policy)
