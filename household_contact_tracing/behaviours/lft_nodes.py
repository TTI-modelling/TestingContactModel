from typing import Callable, List

from household_contact_tracing.network import Node, Network


def lft_nodes(network: Network, time: int, prob_lfa_positive: Callable, params: dict) -> List[Node]:
    """Performs a days worth of lateral flow testing.

    Returns:
        A list of nodes who have tested positive through the lateral flow tests.
    """
    if "node_daily_prob_lfa_test" in params:
        node_daily_prob_lfa_test = params["node_daily_prob_lfa_test"]
    else:
        node_daily_prob_lfa_test = 1

    positive_nodes = []
    for node in network.all_nodes():
        if node.being_lateral_flow_tested:
            if node.will_lfa_test_today(node_daily_prob_lfa_test):
                if not node.received_positive_test_result:
                    if node.lfa_test_node(time, prob_lfa_positive):
                        positive_nodes.append(node)
    return positive_nodes
