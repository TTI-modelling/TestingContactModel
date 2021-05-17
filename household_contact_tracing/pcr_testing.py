from abc import ABC

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

    def receive_pcr_test_results(self, time: int):
        pass

    def pcr_test_node(self, node: Node, time: int):
        pass


class PCRTestingIndividualLevelTracing(PCRTestingBehaviour):

    def receive_pcr_test_results(self, time: int):
        """For nodes who would receive a PCR test result today, update
        """
        # self reporting infections
        [
            self.pcr_test_node(node, time)
            for node in self._network.all_nodes()
            if node.time_of_reporting + node.testing_delay == time
            and not node.contact_traced
            and not node.received_result
        ]

        # contact traced nodes
        [
            self.pcr_test_node(node, time)
            for node in self._network.all_nodes()
            if node.symptom_onset_time + node.testing_delay == time
            and node.contact_traced
            and not node.received_result
        ]

    def pcr_test_node(self, node: Node, time: int):
        """Given a the time relative to a nodes symptom onset, will that node test positive

        Args:
            node (NodeContactModel): The node to be tested today
            time (int): Current time in days
        """
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
        """
        For nodes who would receive a PCR test result today, update
        """

        if self.contact_tracing.lfa_tested_nodes_book_pcr_on_symptom_onset:
           super(PCRTestingIndividualDailyTesting, self).receive_pcr_test_results(time)
        else:
            [
                self.pcr_test_node(node, time)
                for node in self._network.all_nodes()
                if node.time_of_reporting + node.testing_delay == time
                   and not node.received_result
                   and not node.contact_traced
                   and not node.being_lateral_flow_tested
            ]