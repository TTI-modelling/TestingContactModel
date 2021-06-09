from abc import abstractmethod, ABC
from typing import Optional

import numpy

from household_contact_tracing.network import Network, Household


class NewHousehold(ABC):
    def __init__(self, network: Network, hh_propensity_to_use_trace_app,
                 size_mean_contacts_biased_distribution):
        self.network = network
        self.hh_propensity_to_use_trace_app = hh_propensity_to_use_trace_app
        self.size_mean_contacts_biased_distribution = size_mean_contacts_biased_distribution

    @abstractmethod
    def new_household(self, time: int, infected_by: Optional[Household],
                      additional_attributes: Optional[dict] = None) -> Household:
        """Add a new Household to the model."""

    def size_of_household(self) -> int:
        """Generates a random household size

        Returns:
        household_size {int}
        """
        return numpy.random.choice([1, 2, 3, 4, 5, 6],
                                   p=self.size_mean_contacts_biased_distribution)

    def hh_propensity_use_trace_app(self) -> bool:
        if numpy.random.binomial(1, self.hh_propensity_to_use_trace_app) == 1:
            return True
        else:
            return False


class NewHouseholdLevel(NewHousehold):

    def new_household(self, time: int, infected_by: Optional[Household],
                      additional_attributes: Optional[dict] = None) -> Household:
        """Adds a new household to the household dictionary"""
        house_size = self.size_of_household()

        return self.network.add_household(house_size=house_size,
                                          time_infected=time,
                                          infected_by=infected_by,
                                          propensity_trace_app=self.hh_propensity_use_trace_app(),
                                          additional_attributes=additional_attributes
                                          )


class NewHouseholdIndividualTracingDailyTesting(NewHouseholdLevel):

    def new_household(self, time: int, infected_by: Optional[Household],
                      additional_attributes: Optional[dict] = None) -> Household:

        return super().new_household(time, infected_by=infected_by,
                                     additional_attributes={'being_lateral_flow_tested': False,
                                                            'being_lateral_flow_tested_start_time': None,
                                                            'applied_policy_for_household_contacts_of_a_positive_case': False}
                                     )
