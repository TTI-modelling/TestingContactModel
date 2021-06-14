from __future__ import annotations

from household_contact_tracing.network import Network, TestType, PositivePolicy


class HouseholdIsolation:
    def __init__(self, network: Network, household_positive_policy: PositivePolicy):
        self.network = network
        self.household_positive_policy = household_positive_policy

    def isolate_self_reporting_cases(self, time: int):
        """Applies the isolation status to nodes who have reached their self-report time.
        They may of course decide to not adhere to said isolation, or may be a member of a household
        who will not uptake isolation
        """
        for node in self.network.all_nodes():
            if node.will_uptake_isolation:
                if node.time_of_reporting == time:
                    node.isolated = True

    def update_households_contact_traced(self, time: int):
        """Update the contact traced status for all households that have had the
         contact tracing process get there."""
        for household in self.network.all_households:
            if household.time_until_contact_traced <= time:
                if not household.contact_traced:
                    household.update_network()
                    household.isolate_if_symptomatic_nodes(time)
                    household.quarantine_traced_node()

    def update_isolation(self, time: int):
        for node in self.network.all_nodes():
            if node.time_of_reporting + node.testing_delay == time:
                if not node.household.isolated:
                    if not node.household.contact_traced:
                        node.household.isolate_household(time)


class IndividualIsolation(HouseholdIsolation):
    def update_households_contact_traced(self, time: int):
        """Update the contact traced status for all households that have had the
         contact tracing process get there."""
        for household in self.network.all_households:
            if household.time_until_contact_traced <= time:
                if not household.contact_traced:
                    household.update_network()
                    household.quarantine_traced_node()

    def update_isolation(self, time: int):
        for node in self.network.all_nodes():
            if node.time_of_reporting + node.testing_delay == time:
                if node.received_positive_test_result:
                    if not node.household.isolated:
                        if not node.household.contact_traced:
                            node.household.isolate_household(time)


class DailyTestingIsolation(HouseholdIsolation):
    def update_households_contact_traced(self, time: int):
        """Update the contact traced status for all households that have had the
         contact tracing process get there."""
        for household in self.network.all_households:
            if household.time_until_contact_traced <= time:
                if not household.contact_traced:
                    household.update_network()
                    traced_node = household.find_traced_node()
                    # the traced node is now being lateral flow tested
                    if traced_node.node_will_take_up_lfa_testing:
                        if not traced_node.received_positive_test_result:
                            traced_node.being_lateral_flow_tested = True
                            traced_node.time_started_lfa_testing = time

    def update_isolation(self, time: int):
        for node in self.network.all_nodes():
            if node.positive_test_time == time:
                if node.avenue_of_testing == TestType.pcr:
                    if node.received_positive_test_result:
                        if not node.household.applied_household_positive_policy:
                            node.household.apply_positive_policy(time,
                                                                 self.household_positive_policy)
