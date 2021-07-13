import pandas as pd
import pytest
from household_contact_tracing.queueing_processes import Queue, DeterministicQueue


# Testing the Queue object

@pytest.fixture
def simple_Queue():
    """
    Creates a Queue object with some easy parameters for testing.
    """

    my_queue = Queue(days_to_simulate=10, capacity=[10]*10)

    return my_queue


@pytest.fixture
def empty_applicant_df_fixture():
    """
    Loads a fixture of an empty, correctly initialised applicant_df
    """
    return pd.read_pickle('./test/fixtures/queueing_processes/empty_applicant_df.pickle')


def test_Queue_init_applicant_df(simple_queue, empty_applicant_df_fixture):
    return pd.testing.assert_frame_equal(empty_applicant_df_fixture, simple_queue.applicant_df)


@pytest.fixture
def empty_queue_df_fixture():
    """
    Loads a fixture of an empty, correctly initialised applicant_df
    """
    return pd.read_pickle('./test/fixtures/queueing_processes/empty_queue_df.pickle')


def test_Queue_init_queue_df(simple_queue, empty_queue_df_fixture):
    return pd.testing.assert_frame_equal(empty_queue_df_fixture, simple_queue.queue_df)


@pytest.fixture
def Queue_new_applicants_fixture():
    """
    Load a fixture of an applicant df where several applicants have been added
    """
    return pd.read_pickle('./test/fixtures/queueing_processes/Queue_new_applicants_fixture.pickle')


def test_Queue_new_applicants(simple_queue, Queue_new_applicants_fixture):
    simple_queue.add_new_applicants(['A', 'B', 'C'], [1,2,3], 6)

    return pd.testing.assert_frame_equal(Queue_new_applicants_fixture, simple_queue.applicant_df)


@pytest.fixture
def Queue_swab_applicants_fixture():
    """Loads a fixture where some applicants have been processed.
    """
    return pd.read_pickle('./test/fixtures/queueing_processes/Queue_swab_applicants.pickle')

def test_Queue_swab_applicants(simple_queue, Queue_swab_applicants_fixture):
    """Adds some applicants, processes some of the and checks the applicant df
    """
    simple_queue.add_new_applicants(['A', 'B', 'C'], [1,2,3], 6)
    simple_queue.swab_applicants([1, 2], [1,2])

    return pd.testing.assert_frame_equal(Queue_swab_applicants_fixture, simple_queue.applicant_df)


def test_Queue_current_applicants(simple_queue):
    """Checks that the waiting to be processed indexes are returned.

    Add 3 people to the queue, process 2
    """
    simple_queue.add_new_applicants(['A', 'B', 'C'], [1,2,3], 6)
    simple_queue.swab_applicants([1, 2], [1,2])
    assert simple_queue.current_applicants == [0]


def test_Queue_todays_capacity():
    """Checks that the queue returns the right value for todays capacity.
    """
    queue = Queue(days_to_simulate=10, capacity=list(range(10)))

    queue.time = 4

    assert queue.todays_capacity == 4


@pytest.fixture
def DeterministicQueue_add_new_test_seekers_fixture():
    return pd.read_pickle('./test/fixtures/queueing_processes/DeterministicQueue_add_new_test_seekers.pickle')


def test_DeterministicQueue_add_new_test_seekers(DeterministicQueue_add_new_test_seekers_fixture):
    """
    Checks that the add_new_test_seekers method correctly modifies the dataframe
    by adding test seekers based upon the demand
    """
    def processing_delay_dist():
        return 1

    def symptom_onset_delay_dist():
        return 2 

    my_det_queue = DeterministicQueue(
        days_to_simulate            = 10,
        demand                      = [10]*10,
        capacity                    = [10]*10,
        max_time_in_queue           = 10,
        processing_delay_dist  = processing_delay_dist,
        symptom_onset_delay_dist    = symptom_onset_delay_dist
    )

    my_det_queue.add_new_test_seekers()

    return pd.testing.assert_frame_equal(DeterministicQueue_add_new_test_seekers_fixture, my_det_queue.queue.applicant_df)


def test_process_queue_excess_capacity():
    pass

def test_process_queue_excess_demand():
    pass
