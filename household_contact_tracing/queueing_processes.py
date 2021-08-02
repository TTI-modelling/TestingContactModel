'''
Contains queueing process objects that are used to model testing delays and probability of 
not being able to get testing when there are constrained processing resource.
Processing resources can refer to either swabbing capacity, or genetic sequencing capacity.
'''

from datetime import time
import pandas as pd
import numpy as np
import numpy.random as npr
from typing import Callable, List

class Queue:

    def __init__(
            self,
            days_to_simulate: int,
            capacity: List[int]
        ):
        """Creates a queueing process object that stores the current and previous states of the queue.
        Also contains methods for accessing the data in the queue, and changing the data in the queue. 

        Args:
            days_to_simulate (int): The total number of days that will be simulated
            capacity (list): The processing capacity (integer values) at each timepoint
            max_time_in_queue (int): Maximum days from symptom onset to ineligibility for processing
            verbose (bool, optional): If true prints some outputs. Defaults to False.
        """

        self.days_to_simulate   = days_to_simulate
        self.capacity           = capacity

        # default values
        self.time = 0

        # initialise a dataframe that stores summaries of the queue at each timepoint
        self.create_queue_df()
        self.create_applicants_df()


    def create_queue_df(self):
        """
        Queue df stores summaries of the overall status of the queue at each timepoint. It is updated as the calculations progresses
        """

        # create a dataframe to store information about the overall queueing process
        self.queue_df = pd.DataFrame({
            'time':     list(range(self.days_to_simulate)),
            'capacity': self.capacity
        })

        # create some empty columns for storing results
        self.queue_df['new_applicants']                 = ''
        self.queue_df['spillover_to_next_day']          = ''
        self.queue_df['total_applications_today']       = ''
        self.queue_df['capacity_exceeded']              = ''
        self.queue_df['capacity_exceeded_by']           = ''
        self.queue_df['number_processed_today']         = ''
        self.queue_df['number_left_queue_not_tested']   = ''


    def create_applicants_df(self):
        """Applicants df store information about everyone who has applied for a test at each timepoint. It is updated as the calculation progresses.
        """
        
        self.applicant_df = pd.DataFrame()

        # create empty columns for applicants
        self.applicant_df['id']                         = ''
        self.applicant_df['processed']                  = ''
        self.applicant_df['waiting_to_be_processed']    = ''
        self.applicant_df['left_queue_not_processed']   = ''
        self.applicant_df['time_symptom_onset']         = ''
        self.applicant_df['time_joined_queue']          = ''
        self.applicant_df['time_processed']             = ''
        self.applicant_df['time_received_result']       = ''
        self.applicant_df['time_will_leave_queue']      = ''

    def add_new_applicants(
            self,
            ids: list,
            time: int,
            symptom_onset_times: list,
            queue_leaving_times: list
        ):
        """
        Adds new applicants to the queue.
        """

        new_applicant_df = pd.DataFrame(
            {
                'id': ids,
                'time_symptom_onset': symptom_onset_times,
                'time_will_leave_queue': queue_leaving_times
            }
        )

        # initialise other columns with default values
        new_applicant_df['processed']                = False
        new_applicant_df['waiting_to_be_processed']  = True # default value, initially the queue is empty
        new_applicant_df['left_queue_not_processed'] = ''
        new_applicant_df['time_joined_queue']        = time
        new_applicant_df['time_processed']           = ''
        new_applicant_df['time_received_result']     = ''

        self.applicant_df = self.applicant_df.append(new_applicant_df, ignore_index = True)

    def swab_applicants(self, 
        to_be_processed: list, 
        processing_delays: list):
        """For a list of applicants who were successful in getting thorugh the queue, update their variables associated with processing
        Args:
            to_be_processed (list): A list of integers, referring the rows of the applicant_dataframe that will get processed
        """
        
        # The columns that will be updated
        columns_to_update = [
            'waiting_to_be_processed',
            'time_processed',
            'left_queue_not_processed',
            'processed'
        ]

        # record an attribtue of which individuals were processed today for use later
        self.todays_processed_index = to_be_processed

        # update the above status to show they have been processed
        self.applicant_df.loc[to_be_processed, columns_to_update] = [False, self.time, False, True]

        # work out when they receive their result, and update the data
        self.applicant_df.loc[to_be_processed, 'time_received_result'] = self.time + np.array(processing_delays)

        # update the queue_df table with the number of individuals processed today
        self.queue_df.loc[self.queue_df.time == self.time, ['number_processed_today']] = len(to_be_processed)


    def update_queue_leaver_status(self):
        """These individuals have been in the queue too long. They are no longer trying/able to get a swab.
        """

        # These people will leave the queue today
        self.leavers = (self.applicant_df.time_will_leave_queue <= self.time) & (self.applicant_df.waiting_to_be_processed == True)


        # record the number of people who carry over to the next day
        if self.todays_capacity > len(self.current_applicants):
            spillover_to_next_day = 0
        else:
            spillover_to_next_day = len(self.current_applicants) - sum(self.leavers) - self.todays_capacity
        
        #
        self.queue_df.loc[self.time, ['spillover_to_next_day', 'number_left_queue_not_tested']] = [spillover_to_next_day, sum(self.leavers)]

        # Set their waiting to be processed status to False
        self.applicant_df.loc[self.leavers, ['waiting_to_be_processed', 'left_queue_not_processed']] = [False, True]


    @property
    def current_applicants(self) -> list:
        """Gets the indexes of individuals waiting to be processed.

        Returns:
            list: The indexes of individuals waiting to be processed
        """
        return list(self.applicant_df[self.applicant_df.waiting_to_be_processed].index)


    @property
    def todays_capacity(self) -> int:
        """Gets the number of processes that can be performed today.

        Returns:
            int: The number of processes that can be performed today
        """
        return int(self.queue_df[self.queue_df.time == self.time].capacity)


    @property
    def number_processes_performed_today(self) -> int:
        """
        Gets the number of processes that have been completed. This will be the number of new applicants
        or the processing capacity.

        Returns:
            int: The number of processes that have been performed today
        """

        return sum(self.applicant_df.time_processed == self.time)


# controller layout 
class QueueController:

    def __init__(self):
        """
        Queue controllers control how the queue is run. In some
        cases you may want to run the queue on it's own, or you may want
        the queue to be interacting with a branching process model in various ways.
        """

        self.completed = False

    def process_todays_new_demand(self):
        pass

    def process_queue(self):
        pass

    def simulate_one_day(self):
        pass


class DeterministicQueue(QueueController):
    # TODO: Rename, it's not deterministic, but the inputs are

    def __init__(
            self,
            days_to_simulate: int,
            demand: List[int],
            capacity: List[int],
            max_time_in_queue: int,
            processing_delay_dist: Callable,
            symptom_onset_delay_dist: Callable,
            selection_method: str):
        """A simple queueing process object that does not interact with a branching process model.

        The test demand and capacity are pre-determined, and the model works out what happens to the queue.

        Args:
            days_to_simulate (int): Number of simulation steps to be performed
            demand (List[int]): The number of new test seekers at each time step.
            capacity (List[int]): The processing capacity of the queue at each time step.
            max_time_in_queue (int): How long since symptom onset that an individual can remain in the queue 
                                     they become ineligible for testing 
            processing_delay_dist (Callable): A callable that returns integer test processing delays
            symptom_onset_delay_dist (Callable): A callable the returns integer delays of the time from symptom onset to booking a test.
            selection_method ('uniform', 'newest'): Method for selecting which applicants to process when demand exceeds capacity.
        """

        # initialise the queue
        self.queue = Queue(
            days_to_simulate        = days_to_simulate,
            capacity                = capacity
        )

        # set parameters
        self.demand                     = demand
        self.processing_delay_dist      = processing_delay_dist
        self.symptom_onset_delay_dist   = symptom_onset_delay_dist
        self.max_time_in_queue          = max_time_in_queue
        self.days_to_simulate           = days_to_simulate
        self.selection_method           = selection_method

        # ease of acccess stuff
        self.time           = self.queue.time
        

    def add_new_test_seekers(self):
        """
        Adds new test seekers to the queue.

        For this model, the new test seekers at each time point are defined a priori.
        """
        symptom_onset_times = [
                self.time - self.symptom_onset_delay_dist() for _ in range(self.demand[self.time])
        ]

        queue_leaving_times = [
            onset_time + self.max_time_in_queue for onset_time in symptom_onset_times
        ]

        self.queue.add_new_applicants(
            ids = [''] * self.demand[self.time],
            time = self.time,
            symptom_onset_times = symptom_onset_times,
            queue_leaving_times = queue_leaving_times
        )

    def select_applicants_for_processing(self, remaining_processing_capacity: int) -> list:
        """Given the current demand and remaining testing capacity, compute which individuals get selected for testing.

        Args:
            current_queue_applicants (list): A list of id's of individuals who are waiting to get processed.
            remaining_processing_capacity (int): The remaining capacity for individuals to get processed.

        Returns:
            list: The list of processed individuals.
        """
        
        if self.selection_method == 'uniform':
            return(
                npr.choice(
                    a       = self.queue.current_applicants,
                    size    = remaining_processing_capacity,
                    replace = False
                )
            )
        elif self.selection_method == 'newest':
            return(
                self.queue.applicant_df.sort_values('time_joined_queue')[0:remaining_processing_capacity]
            )

    def process_queue(self):
        """
        Performs processing of individuals up to capacity, and updates the dataframes that store the calculations.
        """
        
        # Note: this method is set up so that it can be called multiple times in one day
        # in case new applicants are added multiple times in a day. This is sometimes useful

        number_applicants = len(self.queue.current_applicants)

        # update queue_df with the number of applicants today
        self.queue.queue_df.loc[self.queue.queue_df.time == self.time, ['total_applications_today']] = [number_applicants]

        # how much processing capacity do we have remaining? The method
        remaining_processing_capacity = self.queue.todays_capacity - self.queue.number_processes_performed_today

        # is todays remaining capacity exceeded?
        if number_applicants <= remaining_processing_capacity:
            # if capacity not exceeded, then everyone gets processed

            processing_delays = [
                self.processing_delay_dist() for _ in range(number_applicants)
            ]

            self.queue.swab_applicants(
                to_be_processed = self.queue.current_applicants,
                processing_delays = processing_delays)

        else:
            # Then processing capacity is being exceeded. We process up to capacity.
            # We must select who gets processed, at the moment there is only one method
            # implemented that does this, that picks a subset without replacement

            self.select_applicants_for_processing(remaining_processing_capacity)

            successful_applicants = npr.choice(
                a       = self.queue.current_applicants,
                size    = remaining_processing_capacity,
                replace = False
            )

            processing_delays = [
                self.processing_delay_dist() for _ in range(remaining_processing_capacity)
            ]

            self.queue.swab_applicants(
                to_be_processed = successful_applicants,
                processing_delays = processing_delays)

    def update_queue_leaver_status(self):
        """These individuals have been in the queue too long. They are no longer trying/able to get a swab.
        """

        # These people will leave the queue today
        self.leavers = (self.queue.applicant_df.time_will_leave_queue <= self.time) & (self.queue.applicant_df.waiting_to_be_processed == True)

        self.queue.queue_df.loc[self.time, 'number_left_queue_not_tested'] = [sum(self.leavers)]

        # Set their waiting to be processed status to False
        self.queue.applicant_df.loc[self.leavers, ['waiting_to_be_processed', 'left_queue_not_processed']] = [False, True]

        # work out who will come back the next day
        # not left and not processed
        returners_index = self.queue.applicant_df.waiting_to_be_processed == True

        self.queue.queue_df.loc[self.time, 'spillover_to_next_day'] = [sum(returners_index)]

    def simulate_one_day(self):
        """
        Simulates one day of the queue.
        """

        # steps required to simulate one day
        self.add_new_test_seekers()
        self.update_queue_leaver_status()
        self.process_queue()

        self.queue.time += 1

    def run_simulation(self):
        """Runs the queueing process model.
        """

        while self.time < self.days_to_simulate:

            self.simulate_one_day()

            self.time += 1


class DeterministicQueueVariantSequencing(QueueController):

    def __init__(
            self,
            days_to_simulate: int,
            demand: List[int],
            demand_variant: List[int],
            capacity: List[int],
            max_time_in_queue: int,
            processing_delay_dist: Callable,
            symptom_onset_delay_dist: Callable,
            selection_method: str):
        """A simple queueing process object that does not interact with a branching process model.

        The test demand and capacity are pre-determined, and the model works out what happens to the queue.

        Args:
            days_to_simulate (int): Number of simulation steps to be performed
            demand (List[int]): The number of new test seekers at each time step.
            capacity (List[int]): The processing capacity of the queue at each time step.
            max_time_in_queue (int): How long since symptom onset that an individual can remain in the queue 
                                     they become ineligible for testing 
            processing_delay_dist (Callable): A callable that returns integer test processing delays
            symptom_onset_delay_dist (Callable): A callable the returns integer delays of the time from symptom onset to booking a test.
            selection_method ('uniform', 'newest'): Method for selecting which applicants to process when demand exceeds capacity.
        """

        # initialise the queue
        self.queue = Queue(
            days_to_simulate        = days_to_simulate,
            capacity                = capacity
        )

        # add an empty column to store variant status
        self.queue.applicant_df['variant'] = ''

        # set parameters
        self.demand                     = demand
        self.demand_variant             = demand_variant
        self.processing_delay_dist      = processing_delay_dist
        self.symptom_onset_delay_dist   = symptom_onset_delay_dist
        self.max_time_in_queue          = max_time_in_queue
        self.days_to_simulate           = days_to_simulate
        self.selection_method           = selection_method

        # ease of acccess stuff
        self.time           = self.queue.time
        

    def add_new_queue_joiners(self):
        """
        Adds new test seekers to the queue.

        For this model, the new test seekers at each time point are defined a priori.
        """
        total_new_joiners = self.demand[self.time] + self.demand_variant[self.time]

        symptom_onset_times = [
                self.time - self.symptom_onset_delay_dist() for _ in range(total_new_joiners)
        ]

        queue_leaving_times = [
            onset_time + self.max_time_in_queue for onset_time in symptom_onset_times
        ]

        self.queue.add_new_applicants(
            ids = [''] * total_new_joiners,
            time = self.time,
            symptom_onset_times = symptom_onset_times,
            queue_leaving_times = queue_leaving_times
        )

        # work out which of the new joiners are variants
        variant_ids = npr.choice(
            a = list(range(total_new_joiners)), 
            size = self.demand_variant[self.time], 
            replace = False)

        # by default cases are not variants
        variant_status = [False]*total_new_joiners
        for _ in variant_ids:
            variant_status[_] = True

        # set the variant status on the applicant dataframe column
        todays_joiner_index = self.queue.applicant_df.time_joined_queue == self.time
        self.queue.applicant_df.loc[todays_joiner_index, 'variant'] = variant_status


    def select_applicants_for_processing(self, remaining_processing_capacity: int) -> list:
        """Given the current demand and remaining testing capacity, compute which individuals get selected for testing.

        Args:
            current_queue_applicants (list): A list of id's of individuals who are waiting to get processed.
            remaining_processing_capacity (int): The remaining capacity for individuals to get processed.

        Returns:
            list: The list of processed individuals.
        """
        
        if self.selection_method == 'uniform':
            return(
                npr.choice(
                    a       = self.queue.current_applicants,
                    size    = remaining_processing_capacity,
                    replace = False
                )
            )
        elif self.selection_method == 'newest':
            return(
                self.queue.applicant_df.sort_values('time_joined_queue')[0:remaining_processing_capacity]
            )

    def process_queue(self):
        """
        Performs processing of individuals up to capacity, and updates the dataframes that store the calculations.
        """
        
        # Note: this method is set up so that it can be called multiple times in one day
        # in case new applicants are added multiple times in a day. This is sometimes useful

        number_applicants = len(self.queue.current_applicants)

        # update queue_df with the number of applicants today
        self.queue.queue_df.loc[self.queue.queue_df.time == self.time, ['total_applications_today']] = [number_applicants]

        # how much processing capacity do we have remaining? The method
        remaining_processing_capacity = self.queue.todays_capacity - self.queue.number_processes_performed_today

        # is todays remaining capacity exceeded?
        if number_applicants <= remaining_processing_capacity:
            # if capacity not exceeded, then everyone gets processed

            processing_delays = [
                self.processing_delay_dist() for _ in range(number_applicants)
            ]

            self.queue.swab_applicants(
                to_be_processed = self.queue.current_applicants,
                processing_delays = processing_delays)

        else:
            # Then processing capacity is being exceeded. We process up to capacity.
            # We must select who gets processed, at the moment there is only one method
            # implemented that does this, that picks a subset without replacement

            self.select_applicants_for_processing(remaining_processing_capacity)

            successful_applicants = npr.choice(
                a       = self.queue.current_applicants,
                size    = remaining_processing_capacity,
                replace = False
            )

            processing_delays = [
                self.processing_delay_dist() for _ in range(remaining_processing_capacity)
            ]

            self.queue.swab_applicants(
                to_be_processed = successful_applicants,
                processing_delays = processing_delays)

    def update_queue_leaver_status(self):
        """These individuals have been in the queue too long. They are no longer trying/able to get a swab.
        """

        # These people will leave the queue today
        self.leavers = (self.queue.applicant_df.time_will_leave_queue <= self.time) & (self.queue.applicant_df.waiting_to_be_processed == True)

        self.queue.queue_df.loc[self.time, 'number_left_queue_not_tested'] = [sum(self.leavers)]

        # Set their waiting to be processed status to False
        self.queue.applicant_df.loc[self.leavers, ['waiting_to_be_processed', 'left_queue_not_processed']] = [False, True]

        # work out who will come back the next day
        # not left and not processed
        returners_index = self.queue.applicant_df.waiting_to_be_processed == True

        self.queue.queue_df.loc[self.time, 'spillover_to_next_day'] = [sum(returners_index)]

    def simulate_one_day(self):
        """
        Simulates one day of the queue.
        """

        # steps required to simulate one day
        self.add_new_queue_joiners()
        self.update_queue_leaver_status()
        self.process_queue()

        self.queue.time += 1

    def run_simulation(self):
        """Runs the queueing process model.
        """

        while self.time < self.days_to_simulate:

            self.simulate_one_day()

            self.time += 1


class QueueBranchingProcessController():

    def __init__(
        self,
        queue: Queue):

        self.queue = queue

    def get_todays_queue_output(self):
        """
        Provides outputs from the queueing process that can be passed to a
        branching process model.

        Returns:
            dict: output dict, with the ids and number of processed individuals
        """

        processed_individuals = self.queue.applicant_df.loc[self.queue.todays_processed_index]

        output = {
            'leaving_the_queue_node_ids': self.queue.todays_leavers,
            'processed_individuals': processed_individuals
        }
        
        return output

class QueueAnalyzer():

    def __init__(
        self,
        queue: Queue):

        self.queue = queue
        self.applicant_df = queue.applicant_df
        self.queue_df = queue.queue_df

    def get_prob_getting_processed(self, time_joined_queue: int):
        """
        Returns the probability of getting processed if you join the queue on a specified day
        
        Args:
            time_joined_queue (int): The day of interest
        """
        valid_individuals = (self.queue.applicant_df.time_joined_queue == time_joined_queue) & (self.applicant_df.waiting_to_be_processed == False)
        left_queue_not_processed = self.applicant_df[valid_individuals].left_queue_not_processed
        return 1 - left_queue_not_processed.mean()

    def get_delays_for(self, time_joined_queue: int, delay_from_column: str, delay_to_column: str):
        """
        Return a list of the delays between two timepoints who joined on a specified day
        
        Args:
            time (int): The day on which the applicants joined the queue
            delay_from_column (str): The earliest timepoint
            delay_to_column (str): The latest timepoint
        """
        day_index = (self.applicant_df.time_joined_queue == time_joined_queue) & (self.applicant_df.processed == True)
        delay_from_column = self.applicant_df.loc[day_index, delay_from_column]
        delay_to_column = self.applicant_df.loc[day_index, delay_to_column]
        return delay_to_column - delay_from_column
