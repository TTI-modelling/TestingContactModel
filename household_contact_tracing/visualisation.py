# Code to visualise networks derived from the model
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from network import Node, NodeCollection

contact_traced_edge_colour_within_house = "blue"
contact_traced_edge_between_house = "magenta"
default_edge_colour = "black"
failed_contact_tracing = "red"
app_traced_edge = "green"


def draw_network(nodes: NodeCollection, node_colour: Callable[[Node], str]):
    """Draws the network generated by the model."""

    node_colour_map = [node_colour(node) for node in nodes.all_nodes()]

    # The following chunk of code draws the pretty branching processes
    edge_colour_map = [nodes.G.edges[edge]["colour"] for edge in nodes.G.edges()]

    # Legend for explaining edge colouring
    proxies = [make_proxy(clr, lw=1) for clr in (
            default_edge_colour,
            contact_traced_edge_colour_within_house,
            contact_traced_edge_between_house,
            app_traced_edge,
            failed_contact_tracing
        )
    ]
    labels = (
        "Transmission, yet to be traced",
        "Within household contact tracing",
        "Between household contact tracing",
        "App traced edge",
        "Failed contact trace"
    )

    node_households = {}
    for node in nodes.all_nodes():
        node_households.update({node.node_id: node.household_id})

    # self.pos = graphviz_layout(self.G, prog='twopi')
    plt.figure(figsize=(10, 10))

    nx.draw(
        nodes.G,
        node_size=150, alpha=0.75, node_color=node_colour_map, edge_color=edge_colour_map,
        labels=node_households
    )
    plt.axis('equal')
    plt.title("Household Branching Process with Contact Tracing")
    plt.legend(proxies, labels)


def make_proxy(clr, **kwargs):
    """Used to draw the lines we use in the draw network legend.

    Arguments:
        clr {str} -- the colour of the line to be drawn.

    Returns:
        Line2D -- A Line2D object to be passed to the
    """
    return Line2D([0, 1], [0, 1], color=clr, **kwargs)
