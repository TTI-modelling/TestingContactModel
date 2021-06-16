"""Mappings of colors to node and edge types to be used in views."""
from dataclasses import dataclass

from household_contact_tracing.network.contact_tracing_network import ContactTracingEdgeType, ContactTracingNodeType


@dataclass
class EdgeColour:
    colour: str
    label: str


@dataclass
class NodeColour:
    colour: str
    label: str


edge_colours = {ContactTracingEdgeType.default: EdgeColour("black", "Transmission, yet to be traced"),
                ContactTracingEdgeType.within_house: EdgeColour("blue", "Within household contact tracing"),
                ContactTracingEdgeType.between_house: EdgeColour("magenta",
                                                   "Between household contact tracing"),
                ContactTracingEdgeType.failed_contact_tracing: EdgeColour("red", "Failed contact trace"),
                ContactTracingEdgeType.app_traced: EdgeColour("green", "App traced edge")
                }

node_colours = {ContactTracingNodeType.default: NodeColour("lightgrey", "Default"),
                ContactTracingNodeType.isolated: NodeColour('yellow', "Isolating"),
                ContactTracingNodeType.symptomatic_will_report_infection: NodeColour('lime',
                                                                       "Symptomatic, will report"),
                ContactTracingNodeType.symptomatic_will_not_report_infection: NodeColour('green',
                                                                           "Symptomatic, will not report"),
                ContactTracingNodeType.received_pos_test_pcr: NodeColour('grey', "Received positive PCR"),
                ContactTracingNodeType.received_neg_test_pcr: NodeColour('deeppink', "Received negative PCR"),
                ContactTracingNodeType.confirmatory_pos_pcr_test: NodeColour('turquoise',
                                                               "Positive confirmatory PCR"),
                ContactTracingNodeType.confirmatory_neg_pcr_test: NodeColour('tomato',
                                                               "Negative confirmatory PCR"),
                ContactTracingNodeType.received_pos_test_lfa: NodeColour('pink', "Positive LFA"),
                ContactTracingNodeType.being_lateral_flow_tested_isolated: NodeColour('blue',
                                                                        "Being LFT and isolating"),
                ContactTracingNodeType.being_lateral_flow_tested_not_isolated: NodeColour('orange',
                                                                            "Being LFT and not isolating"),
                ContactTracingNodeType.asymptomatic: NodeColour('olive', 'Asymptomatic')
                }
