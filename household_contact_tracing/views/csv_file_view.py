import os
import pandas as pd
import datetime

from household_contact_tracing.views.simulation_view import SimulationView
from household_contact_tracing.simulation_model import BranchingProcessModel


class CSVFileView(SimulationView):
    """ CSVFile view for storing state change info as a csv file
    """

    def __init__(self, model: BranchingProcessModel, filename=None):
        # Viewers own copies of controller and model (MVC pattern)
        # ... but controller not required yet (no input collected from view)
        # self.controller = controller
        self.model = model

        if filename and os.path.exists(os.path.dirname(self.filename)):
            self._filename = filename
        else:
            self._filename = os.path.join(os.path.dirname(self.model.root_dir),
                                          'temp',
                                          'simulation_output_{}.csv'.format(datetime.datetime.now().strftime("%Y%m%d")))
        # Register as observer
        self.model.register_observer_simulation_stopped(self)

    @property
    def filename(self) -> BranchingProcessModel:
        return self._filename

    @filename.setter
    def filename(self, filename: str):
        if os.path.dirname(filename) and os.path.exists(os.path.dirname(filename)):
            self._filename = filename
        else:
            raise FileNotFoundError('Filename directory {} does not exist'.format(os.path.dirname(filename)))

    def set_display(self, show: bool):
        if show:
            self.model.register_observer_simulation_stopped(self)
        else:
            self.model.remove_observer_simulation_stopped(self)

    def model_state_change(self, subject: BranchingProcessModel):
        """ Respond to changes in model state (e.g. running, extinct, timed-out) """
        pass

    def model_step_increment(self, subject: BranchingProcessModel):
        """ Respond to single step increment in simulation """
        pass

    def model_simulation_stopped(self, subject: BranchingProcessModel):
        """ Respond to end of simulation run """
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
        """ Respond to changes in graph (nodes/households network) """
        pass

