# Code to visualise networks derived from the model

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import networkx as nx

from household_contact_tracing.views.simulation_view import SimulationView
from household_contact_tracing.network.network import Network
from household_contact_tracing.simulation_model import SimulationModel
from household_contact_tracing.views.colors import node_colours, edge_colours


class GraphView(SimulationView):
    """Graph View"""
    def __init__(self, controller, model: SimulationModel):
        # Viewers own copies of controller and model (MVC pattern)
        # ... but controller not required yet (no input collected from view)
        # self.controller = controller
        self.model = model

        # Register as observer
        self.model.register_observer_state_change(self)
        self.model.register_observer_simulation_stopped(self)

    def set_display(self, show: bool):
        if show:
            self.model.register_observer_state_change(self)
            self.model.register_observer_simulation_stopped(self)
        else:
            self.model.remove_observer_state_change(self)
            self.model.remove_observer_simulation_stopped(self)

    def model_param_change(self, subject):
        """ Respond to parameter change(s) """
        pass

    def model_state_change(self, subject):
        """ Respond to changes in model state (e.g. running, extinct, timed-out) """
        pass

    def model_step_increment(self, subject):
        """ Respond to single step increment in simulation """
        pass

    def model_simulation_stopped(self, subject: SimulationModel):
        if self not in subject._observers_graph_change:
            self.draw_network(subject.network)

    def graph_change(self, subject: SimulationModel):
        """ Respond to changes in graph (nodes/households network) """
        self.draw_network(subject.network)

    def set_show_increment_graphs(self, show_all):
        if show_all:
            self.model.register_observer_graph_change(self)
        else:
            self.model.remove_observer_graph_change(self)

    def draw_network(self, network: Network):
        """Draws the network generated by the model."""

        node_colour_map = [node_colours[node.node_type()].colour for node
                           in network.all_nodes()]

        edge_colour_map = [edge_colours[edge].colour for edge in network.edge_types()]

        node_labels = {node.id: node.household.id for node in network.all_nodes()}

        plt.figure(figsize=(10, 10))

        nx.draw(network.graph, node_size=150, alpha=0.75, node_color=node_colour_map,
                edge_color=edge_colour_map, labels=node_labels)
        plt.title("Household Branching Process with Contact Tracing")

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

        # Legend for edge colours
        lines = [Line2D([0, 1], [0, 1], color=clr.colour, lw=1) for clr
                 in edge_colours.values()]
        labels = [value.label for value in edge_colours.values()]
        first_legend = plt.legend(lines, labels, loc="upper left", bbox_to_anchor=(1, 1),
                                  title="Edges")
        plt.gca().add_artist(first_legend)

        # Legend for node colours
        circles = [Circle((0, 0), color=clr.colour, lw=1) for clr in node_colours.values()]
        labels = [value.label for value in node_colours.values()]
        plt.legend(circles, labels, loc="upper left", bbox_to_anchor=(1, 0.85), title="Nodes")

        plt.show()
