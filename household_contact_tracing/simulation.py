from enum import Enum
from typing import List


class EndReason(Enum):
    not_ended = 1
    extinct = 2
    timed_out = 3
    infection_above_threshold = 4


class SimulationResult:
    """The result of a simulation.

    :ivar end_reason: The reason for the simulation ending.
    :ivar died_out: Whether the infection died out when the simulation ended.
    :ivar inf_counts: The number of people infected as a function of time.
    :ivar end_time: The time step on which the simulation ended.
    """

    def __init__(self):
        self.end_reason: EndReason = EndReason.not_ended
        self.died_out: bool = False
        self.inf_counts: List[int] = []
        self.end_time: int = -1

    def end_simulation(self, end_reason: EndReason, end_time: int):
        self.end_reason = end_reason
        self.died_out = end_reason.value == EndReason.extinct
        self.end_time = end_time
