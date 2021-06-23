import datetime
import os

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

        if filename:
            self.filename = filename
        else:
            self.filename = os.path.join(os.path.dirname(self.model.root_dir),
                                         'temp',
                                         'simulation_output_{}.csv'.format(datetime.datetime.now().strftime("%Y%m%d")))

        # Register as observer
        self.model.register_observer_state_change(self)

    def set_display(self, show: bool):
        if show:
            self.model.register_observer_simulation_stopped(self)
        else:
            self.model.remove_observer_simulation_stopped(self)

    def model_param_change(self, subject):
        """ Respond to parameter change(s) """
        pass

    def model_state_change(self, subject):
        """ Respond to changes in model state (e.g. running, extinct, timed-out) """
        pass

    def model_step_increment(self, subject):
        """ Respond to single step increment in simulation """
        pass

    def model_simulation_stopped(self, subject: BranchingProcessModel):
        """ Respond to end of simulation run """
        pass

    def graph_change(self, subject: BranchingProcessModel):
        """ Respond to changes in graph (nodes/households network) """
        pass

