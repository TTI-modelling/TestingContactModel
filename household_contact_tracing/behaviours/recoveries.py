from household_contact_tracing.network.network import Network


def perform_recoveries(network: Network, time: int):
    """
    Loops over all nodes in the branching process and determine recoveries.

    time - The current time of the process, if a nodes recovery time equals the current time, then it is set to the
    recovered state
    """
    for node in network.all_nodes():
        if node.recovery_time == time:
            node.recovered = True
