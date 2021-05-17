"""Methods implementing isolation for each model type.
These methods change the isolation state of Nodes and Households based on various node
and Household attributes."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from household_contact_tracing.contact_tracing import ContactTracing

from household_contact_tracing.network import Network, TestType


class UpdateIsolation(ABC):
    def __init__(self, network: Network):
        self._network = network
        self.contact_tracing = None

    @property
    def contact_tracing(self) -> ContactTracing:
        return self._contact_tracing

    @contact_tracing.setter
    def contact_tracing(self, contact_tracing: ContactTracing):
        self._contact_tracing = contact_tracing

    @abstractmethod
    def update_isolation(self, time):
        """Isolate all non isolated households where the infection has been reported
         (excludes those who will not take up isolation if prob <1)"""
        pass

    def update_all_households_contact_traced(self, time):
        """Update the contact traced status for all households that have had the
         contact tracing process get there."""
        for household in self._network.houses.all_households():
            if household.time_until_contact_traced <= time:
                if not household.contact_traced:
                    self.contact_tracing.contact_trace_household(household, time)


class UpdateIsolationHouseholdLevel(UpdateIsolation):
    def update_isolation(self, time: int):
        self.update_all_households_contact_traced(time)

        for node in self._network.all_nodes():
            if node.time_of_reporting + node.testing_delay == time:
                if not node.household().isolated:
                    if not node.household().contact_traced:
                        self.contact_tracing.contact_trace_household_behaviour.isolate_household(node.household(), time)


class UpdateIsolationIndividualLevelTracing(UpdateIsolation):
    def update_isolation(self, time: int):
        self.update_all_households_contact_traced(time)

        for node in self._network.all_nodes():
            if node.time_of_reporting + node.testing_delay == time:
                if node.received_positive_test_result:
                    if not node.household().isolated:
                        if not node.household().contact_traced:
                            self._contact_tracing.contact_trace_household_behaviour.isolate_household(node.household(), time)


class UpdateIsolationIndividualTracingDailyTesting(UpdateIsolation):
    def update_isolation(self, time: int):
        self.update_all_households_contact_traced(time)

        for node in self._network.all_nodes():
            if node.positive_test_time == time:
                if node.avenue_of_testing == TestType.pcr:
                    if node.received_positive_test_result:
                        if not node.household().applied_policy_for_household_contacts_of_a_positive_case:
                            self.contact_tracing.apply_policy_for_household_contacts_of_a_positive_case(node.household(), time)
