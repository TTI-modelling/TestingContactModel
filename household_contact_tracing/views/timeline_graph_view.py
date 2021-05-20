import matplotlib.pyplot as plt
import pandas as pd

from household_contact_tracing.views.simulation_view import SimulationView
from household_contact_tracing.network import Network, NodeType
from household_contact_tracing.simulation_model import SimulationModel

node_type_colours = {'default': "lightgrey",
                    'isolated': 'yellow',
                    'had_contacts_traced': "orange",
                    'symptomatic_will_report_infection': 'lime',
                    'symptomatic_will_not_report_infection': 'green',
                    'received_pos_test_pcr': 'grey',
                    'received_neg_test_pcr': 'deeppink',
                    'confirmatory_pos_pcr_test': 'turquoise',
                    'confirmatory_neg_pcr_test': 'tomato',
                    'received_pos_test_lfa': 'pink',
                    'being_lateral_flow_tested_isolated': 'blue',
                    'being_lateral_flow_tested_not_isolated': 'orange',
                    'asymptomatic': 'olive'
                    }


class TimelineGraphView(SimulationView):
    """ Timeline View: A really simple proof-of-concept, could be used as a template.
        Shows how views are now decoupled from model code and eachother.
    """

    def __init__(self, controller, model: SimulationModel):
        # Viewers own copies of controller and model (MVC pattern)
        # ... but controller not required yet (no input collected from view)
        # self.controller = controller
        self.model = model

        self.df_node_type_counts = pd.DataFrame(columns=[
                                                    'default',
                                                    'isolated', 'had_contacts_traced',
                                                    'received_pos_test_pcr', 'received_neg_test_pcr',
                                                    'confirmatory_pos_pcr_test', 'confirmatory_neg_pcr_test',
                                                    'received_pos_test_lfa', 'being_lateral_flow_tested_isolated',
                                                    'being_lateral_flow_tested_not_isolated',
                                                    'symptomatic_will_report_infection',
                                                    'symptomatic_will_not_report_infection',
                                                    'asymptomatic',
                                                     ])

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

    def model_step_increment(self, subject: SimulationModel):
        """ Respond to single step increment in simulation """
        self.increment_timeline(subject.network)

    def model_simulation_stopped(self, subject: SimulationModel):
        self.draw_timeline(subject.network)


    def draw_timeline(self, network: Network):
        """ Draws the timeline graph, generated by the model."""

        if len(self.df_node_type_counts.index):
            self.df_node_type_counts.plot(subplots=True, legend=False,
                                               color=node_type_colours, figsize=(3, 10),
                                               ylim=(0, self.df_node_type_counts.to_numpy().max()))
            plt.show()

    def increment_timeline(self, network):
        self.df_node_type_counts = self.df_node_type_counts.append({
            'default': network.count_nodes(NodeType.default),
            'isolated': network.count_nodes(NodeType.isolated),
            'had_contacts_traced': network.count_nodes(NodeType.had_contacts_traced),
            'received_pos_test_pcr': network.count_nodes(NodeType.received_pos_test_pcr),
            'received_neg_test_pcr': network.count_nodes(NodeType.received_neg_test_pcr),
            'confirmatory_pos_pcr_test': network.count_nodes(NodeType.confirmatory_pos_pcr_test),
            'confirmatory_neg_pcr_test': network.count_nodes(NodeType.confirmatory_neg_pcr_test),
            'received_pos_test_lfa': network.count_nodes(NodeType.received_pos_test_lfa),
            'being_lateral_flow_tested_isolated': network.count_nodes(NodeType.being_lateral_flow_tested_isolated),
            'being_lateral_flow_tested_not_isolated':
                network.count_nodes(NodeType.being_lateral_flow_tested_not_isolated),
            'symptomatic_will_report_infection': network.count_nodes(NodeType.symptomatic_will_report_infection),
            'symptomatic_will_not_report_infection':
                network.count_nodes(NodeType.symptomatic_will_not_report_infection),
            'asymptomatic': network.count_nodes(NodeType.asymptomatic)
        }, ignore_index=True)
