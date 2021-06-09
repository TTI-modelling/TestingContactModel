from household_contact_tracing.network import Network


def isolate_self_reporting_cases(network: Network, time: int):
    """Applies the isolation status to nodes who have reached their self-report time.
    They may of course decide to not adhere to said isolation, or may be a member of a household
    who will not uptake isolation
    """
    for node in network.all_nodes():
        if node.will_uptake_isolation:
            if node.time_of_reporting == time:
                node.isolated = True
