from household_contact_tracing.views.branching_process_view import BranchingProcessView
from household_contact_tracing.simulation_model import BranchingProcessModel


class ShellView(BranchingProcessView):
    """
        Shell View: Prints out everything that it is registered to observe (as a string).

        Attributes
        ----------
            _model (BranchingProcessModel): The branching process model who's data is being displayed to the user


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

    def __init__(self, model: BranchingProcessModel):
        """
        Constructor for ShellView

            Parameters:
                model (BranchingProcessModel): The branching process model who's data is being displayed to the user

            Returns:
                new ShellView
        """

        self._model = model

        # Register default observers
        self._model.register_observer_state_change(self)
        self._model.register_observer_simulation_stopped(self)

    def set_display(self, show: bool):
        """
        Sets whether this shell view's outputs (printed string output) are displayed or not.

            Parameters:
                show (bool): To display this view, set to True

            Returns:
                None
        """
        if show:
            self._model.register_observer_graph_change(self)
            self._model.register_observer_state_change(self)
            self._model.register_observer_step_increment(self)
            self._model.register_observer_simulation_stopped(self)
        else:
            self._model.remove_observer_graph_change(self)
            self._model.remove_observer_state_change(self)
            self._model.remove_observer_step_increment(self)
            self._model.remove_observer_simulation_stopped(self)

    def graph_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in graph (nodes/households network)

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        print('Graph changed')

    def model_state_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in model state (e.g. running, extinct, timed-out)

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        print('State change: New state: {}'.format(subject.state))

    def model_step_increment(self, subject: BranchingProcessModel):
        """
        Respond to single step increment in simulation

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        print('Model has been incremented by one step')

    def model_simulation_stopped(self, subject: BranchingProcessModel):
        """
        Respond to end of simulation run

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        print('Simulation has stopped running')
