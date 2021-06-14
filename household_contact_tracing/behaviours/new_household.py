from abc import abstractmethod, ABC
from typing import Optional, List

import numpy

from household_contact_tracing.network import Network, Household
from household_contact_tracing.utilities import update_params


class NewHousehold(ABC):
    def __init__(self, network: Network, params: dict, local_contact_probs: List[float],
                 total_contact_means: List[float]):
        self.network = network
        self.local_contact_probs = local_contact_probs
        self.total_contact_means = total_contact_means
        self.hh_propensity_to_use_trace_app = 1
        self.house_size_probs = [0.294591195, 0.345336927, 0.154070081, 0.139478886,
                                 0.045067385, 0.021455526]

        update_params(self, params)

        # Calculate the expected local contacts
        expected_local_contacts = [self.local_contact_probs[i] * i for i in range(6)]

        # Calculate the expected global contacts
        expected_global_contacts = numpy.array(self.total_contact_means) - numpy.array(expected_local_contacts)

        # Size biased distribution of households (choose a node, what is the prob they are in a house size 6, this is
        # biased by the size of the house)
        size_mean_contacts_biased_distribution = [(i + 1) * self.house_size_probs[i] * expected_global_contacts[i] for i
                                                  in range(6)]
        total = sum(size_mean_contacts_biased_distribution)
        self.size_mean_contacts_biased_distribution = [prob / total for prob in size_mean_contacts_biased_distribution]

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
                                                            'applied_household_positive_policy': False}
                                     )
