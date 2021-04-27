from household_contact_tracing.simulation_controller_interface import SimulationControllerInterface
from household_contact_tracing.simulation_model_interface import SimulationModelInterface
from household_contact_tracing.shell_view import ShellView
from household_contact_tracing.graph_view import GraphView


class SimulationController(SimulationControllerInterface):
    """ Branching Process Simulation Controller (MVC pattern) """

    def __init__(self, model: SimulationModelInterface):
        self._model = model
        self.shellView = ShellView(self, model)
        self.graphView = GraphView(self, model)

    def reset(self):
        """ Reset the simulation model."""
        self._model.initialise_simulation()

    def run_simulation(self, num_steps: int = 20, infection_threshold: int = 5000):
        """ Run the simulation."""
        self._model.run_simulation(num_steps, infection_threshold)

    def set_params(self, params):
        """ Set parameters for the simulation."""
        self._model.set_params(params)
