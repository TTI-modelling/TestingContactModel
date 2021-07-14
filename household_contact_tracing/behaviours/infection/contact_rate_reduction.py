from abc import ABC, abstractmethod

from household_contact_tracing.network import Node
from household_contact_tracing.parameterised import Parameterised


class ContactRateReduction(ABC, Parameterised):
    """
        An abstract base class used to represent the highest level 'Contact Rate Reduction' behaviour.

        Note:   This class forms part of a 'Strategy' pattern. All child classes implement a family of possible
                behaviours or strategies (ways of obtaining a contact rate reduction).
                Add further child classes to add new behaviour types (strategies) that can be selected and updated at
                design or run-time.

        Attributes
        ----------
            reduce_contacts_by
                Todo
            global_contact_reduction_imperfect_quarantine
                Todo
            global_contact_reduction_risky_behaviour
                todo

        Methods
        -------

        get_contact_rate_reduction(self, node) -> int
            Returns a contact rate reduction, depending upon a nodes current status and various
            intervention parameters

    """

    def __init__(self, params: dict):
        self.reduce_contacts_by = 0
        self.global_contact_reduction_imperfect_quarantine = 0
        self.global_contact_reduction_risky_behaviour = 0

        self.update_params(params)

    @abstractmethod
    def get_contact_rate_reduction(self, node: Node) -> int:
        """
        Returns a contact rate reduction, depending upon a nodes current status and various
        intervention parameters

        Parameters
        -------
        :param node (ContactTracingNode)
            The node that is having its contact rate reduction calculated
        """


class ContactRateReductionHouseholdLevelTracing(ContactRateReduction):

    def get_contact_rate_reduction(self, node: Node) -> int:
        """Returns a contact rate reduction, depending upon a nodes current status and various
        intervention parameters
        """

        if node.isolated and node.propensity_imperfect_isolation:
            return self.global_contact_reduction_imperfect_quarantine
        elif node.isolated and not node.propensity_imperfect_isolation:
            # return 1 means 100% of contacts are stopped
            return 1
        else:
            return self.reduce_contacts_by


class ContactRateReductionIndividualTracingDaily(ContactRateReduction):

    def get_contact_rate_reduction(self, node: Node) -> int:
        """This method overrides the default behaviour. Previously the override behaviour allowed
        he global contact reduction to vary by household size.

        We override this behaviour, so that we can vary the global contact reduction by whether a
        node is isolating or being lfa tested or whether they engage in risky behaviour while they
         are being lfa tested.

        Remember that a contact rate reduction of 1 implies that 100% of contacts are stopped.
        """
        # the isolated status should never apply to an individual who will not uptake intervention

        if node.isolated and not node.propensity_imperfect_isolation:
            # perfect intervention
            return 1

        elif node.isolated and node.propensity_imperfect_isolation:
            # imperfect intervention
            return self.global_contact_reduction_imperfect_quarantine

        elif node.being_lateral_flow_tested and node.propensity_risky_behaviour_lfa_testing:
            # engaging in risky behaviour while testing negative
            return self.global_contact_reduction_risky_behaviour

        else:
            # normal levels of social distancing
            return self.reduce_contacts_by
