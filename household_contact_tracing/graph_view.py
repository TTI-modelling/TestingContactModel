# Code to visualise networks derived from the model
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from household_contact_tracing.simulation_view_interface import SimulationViewInterface
from household_contact_tracing.network import Node, Network
from household_contact_tracing.simulation_model_interface import SimulationModelInterface

contact_traced_edge_colour_within_house = "blue"
contact_traced_edge_between_house = "magenta"
default_edge_colour = "black"
failed_contact_tracing = "red"
app_traced_edge = "green"


def make_proxy(clr, **kwargs):
    """Used to draw the lines we use in the draw network legend.

    Arguments:
        clr {str} -- the colour of the line to be drawn.

    Returns:
        Line2D -- A Line2D object to be passed to the
    """
    return Line2D([0, 1], [0, 1], color=clr, **kwargs)


class GraphView(SimulationViewInterface):
    '''
        Graph View
    '''

    edge_type_colours = {'within_house': "blue",
                         'between_house': "magenta",
                         'default': "black",
                         'failed_contact_tracing': "red",
                         'app_traced': "green"}

    node_type_colours = {'default': 'white',
                         'isolated': 'yellow',
                         'had_contacts_traced': 'orange',
                         'symptomatic_will_report_infection': 'lime',
                         'symptomatic_will_not_report_infection': 'green',

                         'received_pos_test_pcr': 'grey', 'received_neg_test_pcr': 'deeppink',
                         'confirmatory_pos_pcr_test': 'turquoise', 'confirmatory_neg_pcr_test': 'tomato',
                         'received_pos_test_lfa': 'pink', 'being_lateral_flow_tested_isolated': 'blue',
                         'being_lateral_flow_tested_not_isolated': 'orange'}

    def __init__(self, controller, model: SimulationModelInterface):
        # Viewers own copies of controller and model (MVC pattern)
        # ... but controller not required yet (no input collected from view)
        #self.controller = controller
        self.model = model

        # Register as observer
        self.model.register_observer_state_change(self)
        self.model.register_observer_simulation_stopped(self)

    def set_show_all_graphs(self, show_all):
        if show_all:
            self.model.register_observer_graph_change(self)
        else:
            self.model.remove_observer_graph_change(self)

    def model_simulation_stopped(self, subject: SimulationModelInterface):
        print('graph view observed that simulation has stopped running')
        if self not in subject._observers_graph_change:
            self.draw_network(subject.network)

    def graph_change(self, subject: SimulationModelInterface):
        """ Respond to changes in graph (nodes/households network) """
        print('graph view observed that graph changed')
        self.draw_network(subject.network)

    def draw_network(self, network: Network):
        """Draws the network generated by the model."""

        node_colour_map = [self.node_type_colours[node.node_type()] for node in network.all_nodes()]

        # The following chunk of code draws the pretty branching processes
        edge_colour_map = [self.edge_type_colours[network.graph.edges[edge]["edge_type"]] for edge in network.graph.edges()]

        # Legend for explaining edge colouring
        proxies = [
            make_proxy(clr, lw=1) for clr in (
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
        for node in network.all_nodes():
            node_households.update({node.node_id: node.household_id})

        # self.pos = graphviz_layout(self.G, prog='twopi')
        plt.figure(figsize=(10, 10))

        nx.draw(
            network.graph,
            node_size=150, alpha=0.75, node_color=node_colour_map, edge_color=edge_colour_map,
            labels=node_households
        )
        plt.axis('equal')
        plt.title("Household Branching Process with Contact Tracing")
        plt.legend(proxies, labels)
        plt.show()
