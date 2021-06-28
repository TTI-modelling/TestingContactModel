import os
import pandas as pd
import datetime

from household_contact_tracing.views.branching_process_view import BranchingProcessView
from household_contact_tracing.simulation_model import BranchingProcessModel


class CSVFileView(BranchingProcessView):
    """
        CSV file view for storing state change info as a csv file
        Shows how views are now decoupled from model code and each other.

        Attributes
        ----------
            _model (BranchingProcessModel):
                The branching process model who's data is being displayed to the user
            _filename (str):
                The file spec used to save the output file.


        Methods
        -------

            set_display(self, display: bool)
                choose whether to show these 'shell' (text printouts) to the user

            graph_change(self, subject: BranchingProcessModel)
                Respond to changes in graph (nodes/households network)

            model_state_change(self, subject: BranchingProcessModel):
                Respond to changes in model state (e.g. running, extinct, timed-out)

            model_step_increment(self, subject: BranchingProcessModel):
                Respond to increment in simulation

            model_simulation_stopped(self, subject: BranchingProcessModel)
                Respond to end of simulation run

    """

    def __init__(self, model: BranchingProcessModel, filename=None):
        """
        Constructor for CSVFileView

            Parameters:
                model (BranchingProcessModel):
                                The branching process model who's data is being displayed to the user

                filename (str): The file spec used to save the output file.
                                If None, the default filename is used: the project temp directory

            Returns:
                new CSVFileView
        """

        self._model = model

        # Initialise the filename, used to save the output file
        if filename:
            self._filename = filename
        else:
            self._filename = os.path.join(os.path.dirname(self._model.root_dir),
                                          'temp',
                                          'simulation_output_{}.csv'.format(datetime.datetime.now().strftime("%Y%m%d")))
        # Register as observer
        self._model.register_observer_simulation_stopped(self)

    @property
    def filename(self) -> str:
        """ Get filename (the file spec used to save the output file). """
        return self._filename

    @filename.setter
    def filename(self, filename: str):
        """ Set filename (the file spec used to save the output file).
            Checks whether directory path exists and raises IsADirectoryError if not
        """
        if os.path.dirname(filename) and os.path.exists(os.path.dirname(filename)):
            self._filename = filename
        # Todo check file name is (without directory path) is valid.
        else:
            raise IsADirectoryError('Directory {} does not exist'.format(os.path.dirname(filename)))

    def set_display(self, show: bool):
        """
        Sets whether this csv file view is created or not.

            Parameters:
                show (bool): To create this view, set to True

            Returns:
                None
        """
        if show:
            self._model.register_observer_simulation_stopped(self)
        else:
            self._model.remove_observer_simulation_stopped(self)

    def model_state_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in model state (e.g. running, extinct, timed-out)

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    def model_step_increment(self, subject: BranchingProcessModel):
        """
        Respond to single step increment in simulation

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        pass

    def model_simulation_stopped(self, subject: BranchingProcessModel):
        """
        Respond to end of simulation run

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        # Add state info to CSV file

        dict_flattened = {'run_finished': str(datetime.datetime.now()),
                          'end_state': subject.state.name}
        for key in subject.state.info:
            dict_flattened[key] = [subject.state.info[key]]

        df_new_state = pd.DataFrame.from_dict(data=dict_flattened)

        # Check if file exists and if so read contents to dataframe, if not, create new dataframe
        try:
            df_history_states = pd.read_csv(self._filename)
        except FileNotFoundError:
            df_history_states = pd.DataFrame()
        df_history_states = pd.concat([df_history_states, df_new_state])
        df_history_states.to_csv(self._filename, index=False)
        print('Added final run results to file: {}'.format(self._filename))

    def graph_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in graph (nodes/households network)

            Parameters:
                subject (SimulationModel): The simulation model being displayed by this simulation view.

            Returns:
                None
        """
        pass
