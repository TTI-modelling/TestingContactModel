# Code to visualise networks derived from the model

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import networkx as nx

from household_contact_tracing.views.simulation_view import SimulationView
from household_contact_tracing.network import Network
from household_contact_tracing.simulation_model import BranchingProcessModel
from household_contact_tracing.views.colors import node_colours, edge_colours


class GraphView(SimulationView):
    """
        Graph view for visually displaying network as a pyplot graph.

        Attributes
        ----------
            _model (BranchingProcessModel): The branching process model who's data is being displayed to the user

        Methods
        -------

            set_display(self, display: bool)
                choose whether to show these 'shell' (text printouts) to the user

            set_show_increment_graphs(self, show_all):
                Sets whether to display this graph view every time the graph is incremented.

            draw_network(network: Network) (Static)
                Draws and shows the network generated by the model using pyplot plot() and show().

            graph_change(self, subject: BranchingProcessModel)
                Respond to changes in graph (nodes/households network)

            model_state_change(self, subject: BranchingProcessModel):
                Respond to changes in model state (e.g. running, extinct, timed-out)

            model_step_increment(self, subject: BranchingProcessModel):
                Respond to increment in simulation

            model_simulation_stopped(self, subject: BranchingProcessModel)
                Respond to end of simulation run

    """
    def __init__(self, model: BranchingProcessModel):
        """
        Constructor for GraphView

            Parameters:
                model (BranchingProcessModel): The branching process model who's data is being displayed to the user

            Returns:
                new GraphView
        """
        self._model = model

        # Register as observer
        self._model.register_observer_state_change(self)
        self._model.register_observer_simulation_stopped(self)

    def set_display(self, show: bool):
        """
        Sets whether this graph view is displayed or not.

            Parameters:
                show (bool): To display this view, set to True

            Returns:
                None
        """
        if show:
            self._model.register_observer_state_change(self)
            self._model.register_observer_simulation_stopped(self)
        else:
            self._model.remove_observer_state_change(self)
            self._model.remove_observer_simulation_stopped(self)

    def set_show_increment_graphs(self, show_all):
        """
        Sets whether to display this graph view every time the graph is incremented.

            Parameters:
                show_all (bool): To display this view, set to True

            Returns:
                None
        """
        if show_all:
            self._model.register_observer_graph_change(self)
        else:
            self._model.remove_observer_graph_change(self)

    @staticmethod
    def draw_network(network: Network):
        """
        Draws and shows the network generated by the model using pyplot plot() and show().

            Parameters:
                network (Network): The network to be drawn and displayed.

            Returns:
                None
        """

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

    def model_state_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in model state (e.g. running, extinct, timed-out)

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    def model_step_increment(self, subject: BranchingProcessModel):
        """
        Respond to single step increment in simulation

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    def model_simulation_stopped(self, subject: BranchingProcessModel):
        """
        Respond to end of simulation run

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        if self not in subject.observers_graph_change:
            self.draw_network(subject.network)

    def graph_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in graph (nodes/households network)

            Parameters:
                subject (SimulationModel): The simulation model being displayed by this simulation view.

            Returns:
                None
        """
        self.draw_network(subject.network)
