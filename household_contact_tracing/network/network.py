from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Iterator, List, Tuple, Dict, Callable
import networkx as nx
from enum import Enum


class EdgeType(Enum):
    pass


class NodeType(Enum):
    pass


class Network(ABC):
    """
        A class used to store data in a graph/network format with nodes and their connecting edges.
        Uses networkx as storage tool.

        Attributes
        ----------
        graph (nx.Graph)
            the persistent storage of graph data

    """
    def __init__(self):
        # Call superclass constructor
        self.graph = nx.Graph()

    @property
    def node_count(self):
        return nx.number_of_nodes(self.graph)

    def is_isomorphic(self, network: Network) -> bool:
        """ Determine whether graphs have identical network structures."""
        return nx.is_isomorphic(self.graph, network.graph)

    def __eq__(self, other):
        """ Currently only determines whether graphs have identical network structures,
            but we may want to compare more details.
        """
        return self.is_isomorphic(other)

    def node(self, node_id: int) -> Node:
        """Return the Node from the Network with `node_id`."""
        return self.graph.nodes[node_id]['node_obj']

    def all_nodes(self) -> Iterator[Node]:
        """Return a list of all nodes in the Network"""
        return (self.node(n) for n in self.graph)

    def count_nodes(self, node_type: NodeType) -> int:
        """Returns the number of nodes of type `node_type`."""
        return sum([node.node_type() == node_type for node in self.all_nodes()])

class Node:
    """
        A class used to store the nodes of a graph/network format.
        Uses networkx as storage tool.

        Attributes
        ----------
        id (int)
            the id of the node

        Methods
        -------
        node_type(self, time=None) -> NodeType:
            Returns a node type, given the current status of the node.

    """
    def __init__(self, node_id: int):
        self.id = node_id

    @abstractmethod
    def node_type(self, time=None) -> NodeType:
        '''
            Returns a node type, given the current status of the node.

            params
                time (int): The current increment / step number (e.g. day number) of the simulation
        '''