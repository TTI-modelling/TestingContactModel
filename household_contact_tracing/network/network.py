from __future__ import annotations
from abc import ABC
from typing import Optional, Iterator, List, Tuple, Dict, Callable
import networkx as nx

from household_contact_tracing.utilities import update_params


class Network(ABC):
    def __init__(self):
        # Call superclass constructor
        self.graph = nx.Graph()

class Node:
    def __init__(self, node_id: int):
        self.id = node_id