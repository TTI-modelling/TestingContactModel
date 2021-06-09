from abc import ABC, abstractmethod

from household_contact_tracing.utilities import update_params


class ContactRateReduction(ABC):
    def __init__(self, params: dict):

        self.reduce_contacts_by = 0
        self.global_contact_reduction_imperfect_quarantine = 0
        self.global_contact_reduction_risky_behaviour = 0

        update_params(self, params)

    @abstractmethod
    def get_contact_rate_reduction(self, node) -> int:
        """Returns a contact rate reduction, depending upon a nodes current status and various
        isolation parameters"""


class ContactRateReductionHouseholdLevelTracing(ContactRateReduction):

    def get_contact_rate_reduction(self, node) -> int:
        """Returns a contact rate reduction, depending upon a nodes current status and various
        isolation parameters
        """

        if node.isolated and node.propensity_imperfect_isolation:
            return self.global_contact_reduction_imperfect_quarantine
        elif node.isolated and not node.propensity_imperfect_isolation:
            # return 1 means 100% of contacts are stopped
            return 1
        else:
            return self.reduce_contacts_by


class ContactRateReductionIndividualTracingDaily(ContactRateReduction):

    def get_contact_rate_reduction(self, node) -> int:
        """This method overrides the default behaviour. Previously the override behaviour allowed
        he global contact reduction to vary by household size.

        We override this behaviour, so that we can vary the global contact reduction by whether a
        node is isolating or being lfa tested or whether they engage in risky behaviour while they
         are being lfa tested.

        Remember that a contact rate reduction of 1 implies that 100% of contacts are stopped.
        """
        # the isolated status should never apply to an individual who will not uptake isolation

        if node.isolated and not node.propensity_imperfect_isolation:
            # perfect isolation
            return 1

        elif node.isolated and node.propensity_imperfect_isolation:
            # imperfect isolation
            return self.global_contact_reduction_imperfect_quarantine

        elif node.being_lateral_flow_tested and node.propensity_risky_behaviour_lfa_testing:
            # engaging in risky behaviour while testing negative
            return self.global_contact_reduction_risky_behaviour

        else:
            # normal levels of social distancing
            return self.reduce_contacts_by
