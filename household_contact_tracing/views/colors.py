"""Mappings of colors to node and edge types to be used in views."""
from dataclasses import dataclass

from household_contact_tracing.network import EdgeType, NodeType


@dataclass
class EdgeColour:
    colour: str
    label: str


@dataclass
class NodeColour:
    colour: str
    label: str


edge_colours = {EdgeType.default: EdgeColour("black", "Transmission, yet to be traced"),
                EdgeType.within_house: EdgeColour("blue", "Within household contact tracing"),
                EdgeType.between_house: EdgeColour("magenta",
                                                   "Between household contact tracing"),
                EdgeType.failed_contact_tracing: EdgeColour("red", "Failed contact trace"),
                EdgeType.app_traced: EdgeColour("green", "App traced edge")
                }

node_colours = {NodeType.default: NodeColour("lightgrey", "Default"),
                NodeType.isolated: NodeColour('yellow', "Isolating"),
                NodeType.symptomatic_will_report_infection: NodeColour('lime',
                                                                       "Symptomatic, will report"),
                NodeType.symptomatic_will_not_report_infection: NodeColour('green',
                                                                           "Symptomatic, will not report"),
                NodeType.received_pos_test_pcr: NodeColour('grey', "Received positive PCR"),
                NodeType.received_neg_test_pcr: NodeColour('deeppink', "Received negative PCR"),
                NodeType.confirmatory_pos_pcr_test: NodeColour('turquoise',
                                                               "Positive confirmatory PCR"),
                NodeType.confirmatory_neg_pcr_test: NodeColour('tomato',
                                                               "Negative confirmatory PCR"),
                NodeType.received_pos_test_lfa: NodeColour('pink', "Positive LFA"),
                NodeType.being_lateral_flow_tested_isolated: NodeColour('blue',
                                                                        "Being LFT and isolating"),
                NodeType.being_lateral_flow_tested_not_isolated: NodeColour('orange',
                                                                            "Being LFT and not isolating"),
                NodeType.asymptomatic: NodeColour('olive', 'Asymptomatic')
                }
