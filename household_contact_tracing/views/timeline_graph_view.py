import math

import matplotlib.pyplot as plt
import pandas as pd

from household_contact_tracing.views.simulation_view import SimulationView
from household_contact_tracing.network import NodeType
from household_contact_tracing.simulation_model import SimulationModel, BranchingProcessModel
from household_contact_tracing.views.colors import node_colours


class TimelineGraphView(SimulationView):
    """ Timeline View: A really simple proof-of-concept, could be used as a template.
        Shows how views are now decoupled from model code and each other.
    """

    def __init__(self, model: BranchingProcessModel):
        # Viewers own copies of controller and model (MVC pattern)
        # ... but controller not required yet (no input collected from view)
        # self.controller = controller
        self.model = model

        self.node_type_counts = pd.DataFrame(columns=[node.name for node in NodeType])

        # Register as observer
        self.model.register_observer_simulation_stopped(self)
        self.model.register_observer_step_increment(self)

    def set_display(self, show: bool):
        if show:
            self.model.register_observer_simulation_stopped(self)
            self.model.register_observer_step_increment(self)
        else:
            self.model.remove_observer_graph_change(self)
            self.model.remove_observer_step_increment(self)

    def model_param_change(self, subject: SimulationModel):
        """ Respond to parameter change(s) """
        pass

    def model_state_change(self, subject: SimulationModel):
        """ Respond to changes in model state (e.g. running, extinct, timed-out) """
        pass

    def graph_change(self, subject: SimulationModel):
        """ Respond to changes in graph (nodes/households network) """
        pass

    def model_step_increment(self, subject: BranchingProcessModel):
        """ Respond to single step increment in simulation """
        self.increment_timeline(subject.network)

    def model_simulation_stopped(self, subject: SimulationModel):
        self.draw_timeline()

    def draw_timeline(self):
        """ Draws the timeline graph, generated by the model."""
        colours = {node.name: colour.colour for node, colour in node_colours.items()}
        if len(self.node_type_counts.index):
            axes = self.node_type_counts.plot(subplots=True, legend=False,
                                              color=colours, figsize=(6, 14),
                                              layout=(math.ceil(len(colours)/2), 2),
                                              ylim=(0, self.node_type_counts.to_numpy().max()))
            for index, label in enumerate(self.node_type_counts.columns):
                title = node_colours[NodeType[label]].label
                axes[index // 2][index % 2].set_title(title)
            plt.tight_layout()
            plt.show()

    def increment_timeline(self, network):
        node_counts = {node.name: network.count_nodes(node) for node in NodeType}
        self.node_type_counts = self.node_type_counts.append(node_counts, ignore_index=True)
