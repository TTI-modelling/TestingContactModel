"""These methods implement PCR testing for the various models."""

from abc import ABC, abstractmethod

import numpy as np

from household_contact_tracing.contact_tracing import ContactTracing
from household_contact_tracing.network import Network, Node, TestType


class PCRTestingBehaviour(ABC):
    def __init__(self, network: Network):
        self._network = network
        self._contact_tracing = None

    @property
    def contact_tracing(self) -> ContactTracing:
        return self._contact_tracing

    @contact_tracing.setter
    def contact_tracing(self, contact_tracing: ContactTracing):
        self._contact_tracing = contact_tracing

    @abstractmethod
    def receive_pcr_test_results(self, time: int):
        """For nodes who would receive a PCR test result today, update."""

    @abstractmethod
    def pcr_test_node(self, node: Node, time: int):
        """Given a the time relative to a nodes symptom onset, will that node test positive

           Args:
               node: The node to be tested today
               time: Current time in days
        """


class PCRTestingIndividualLevelTracing(PCRTestingBehaviour):

    def receive_pcr_test_results(self, time: int):
        # self reporting infections
        for node in self._network.all_nodes():
            if node.time_of_reporting + node.testing_delay == time:
                if not node.contact_traced:
                    if not node.received_result:
                        self.pcr_test_node(node, time)

        # contact traced nodes
        for node in self._network.all_nodes():
            if node.symptom_onset_time + node.testing_delay == time:
                if node.contact_traced:
                    if not node.received_result:
                        self.pcr_test_node(node, time)

    def pcr_test_node(self, node: Node, time: int):
        node.received_result = True
        infectious_age_when_tested = time - node.testing_delay - node.time_infected
        prob_positive_result = self.contact_tracing.prob_testing_positive_pcr_func(infectious_age_when_tested)
        node.avenue_of_testing = TestType.pcr

        if np.random.binomial(1, prob_positive_result) == 1:
            node.received_positive_test_result = True
            node.positive_test_time = time
        else:
            node.received_positive_test_result = False


class PCRTestingIndividualDailyTesting(PCRTestingIndividualLevelTracing):

    def receive_pcr_test_results(self, time: int):
        """For nodes who would receive a PCR test result today, update"""

        if self.contact_tracing.lfa_tested_nodes_book_pcr_on_symptom_onset:
            super().receive_pcr_test_results(time)
        else:
            for node in self._network.all_nodes():
                if node.time_of_reporting + node.testing_delay == time:
                    if not node.contact_traced:
                        if not node.received_result:
                            if not node.being_lateral_flow_tested:
                                self.pcr_test_node(node, time)
