"""
Code that deals with common hyperparameter optimisation routines.

Often, we want to calibrate an epidemic to a desired growth rate and household secondary attack rate.
"""

from abc import ABC
from household_contact_tracing.branching_process_models import HouseholdLevelTracing
from household_contact_tracing.branching_process_controller import BranchingProcessController
from ax import optimize
from copy import Error, copy


class Calibration(ABC):
    """
    Base class for hyperparameter optimisation of infection dynamics.
    """
    
    def __init__(self) -> None:
        pass

    def setup_and_run_model(self):
        pass

    def compute_evaluation(self) -> float:
        pass

    def optimize(self):
        pass

    def plot_results(self):
        pass

    def evaluate_fit(self):
        pass

class StandardCalibrationHouseholdLevelTracing(Calibration):
    """Our standard calibration of HouseholdLevelTracing tunes the models growth rate and 
    household secondary attack rate given inputs: asymptomatic prob, asymptomatic relative infectiousness,
    symptom reporting probability.

    The calibration is carried out in the absence of contact tracing to define a baseline epidemic. Contact tracing
    can then be evaluated against the baseline epidemic.
    """
    
    def __init__(
            self,
            #household_pairwise_survival_prob: float,
            desired_growth_rate: float,
            asymptomatic_prob: float,
            asymptomatic_relative_infectivity: float,
            infection_reporting_prob: float,
            reduce_contacts_by: float,
            starting_infections: int = 100
            ):

        # initialise non-infection parameters that are held constant between simulations
        self.fixed_params = {
            'household_pairwise_survival_prob': 0.2,
            'contact_tracing_success_prob': 0.0,
            'overdispersion': 0.32,
            'infection_reporting_prob': 0.25,
            'contact_trace': False,
            'test_delay': 2,
            'contact_trace_delay': 1,
            'incubation_period_delay': 5,
            'symptom_reporting_delay': 1,
            'do_2_step': False,
            'reduce_contacts_by': 0.6,
            'prob_has_trace_app': 0,
            'hh_propensity_to_use_trace_app': 1,
            'test_before_propagate_tracing': True,
            'starting_infections': 100, 
            'node_will_uptake_isolation_prob': 1,
            'self_isolation_duration': 0,
            'quarantine_duration': 0,
            'transmission_probability_multiplier': 1,
            'propensity_imperfect_quarantine': 0,
            'global_contact_reduction_imperfect_quarantine': 0
        }

        # set the inputted defaults
        self.fixed_params['asymptomatic_prob']                  = asymptomatic_prob
        self.fixed_params['asymptomatic_relative_infectivity']  = asymptomatic_relative_infectivity
        self.fixed_params['infection_reporting_prob']           = infection_reporting_prob
        self.fixed_params['reduce_contacts_by']                 = reduce_contacts_by
        self.fixed_params['starting_infections']                = starting_infections

        self.desired_growth_rate = desired_growth_rate

        self.optimisation_complete = False

    def eval_growth_rate(
            self, 
            outside_household_infectivity_scaling: float,
            max_time: int = 20,
            max_active_infections: int = 1e5) -> float:
        """Sets up a model, runs it, and returns the evaluated growth rate.

        Args:
            outside_household_infectivity_scaling (float): controls how infectious global contacts are
            max_time (int, optional): upper limit of days to simulate. Defaults to 20.
            max_active_infections (int, optional): simulation ends early if maximum number of infections is exceeded. Defaults to 1e5.
        """

        params = copy(self.fixed_params)
        params['outside_household_infectivity_scaling'] = outside_household_infectivity_scaling
        
        controller = BranchingProcessController(HouseholdLevelTracing(params))

        controller.run_simulation(max_time, max_active_infections)

        return controller.growth_rate_view.get_growth_rate()

    def evaluate_fit(self, outside_household_infectivity_scaling) -> float:
        
        return abs(self.desired_growth_rate - self.eval_growth_rate(outside_household_infectivity_scaling))

    def optimise(self, 
            outside_household_infectivity_scaling_range: list[float],
            total_trials: int = 20):
        """Performs the hyperparameter optimization step with proposals from the specified ranges.

        Args:
            outside_household_infectivity_scaling_range (list[float]): The lower and upper values for this parameter.
            total_trials (int): The total number of trials to perform. Defaults to 20
        """

        self.best_parameters, self.values, self.experiment, self.model = optimize(
            parameters=[
                {
                    "name": "outside_household_infectivity_scaling",
                    "type": "range",
                    "bounds": outside_household_infectivity_scaling_range,
                }
            ],
            evaluation_function = lambda p: self.evaluate_fit(p["outside_household_infectivity_scaling"]),
            minimize            = True,
            total_trials        = total_trials
        )
        
        self.optimisation_complete  = True

        return self.best_parameters, self.values

    def get_fitted_growth_rate_samples(
        self,
        n_obs: int = 10) -> list[float]:
        """If optimisation has been completed, this method generates sample of the growth rate using
        the results from the optimisation step.


        Args:
            n_obs (int, optional): Number of fitted sample of the growth rate to get. Defaults to 10.

        Returns:
            [list]: A list containing fitted samples of the growth rate.
        """

        if self.optimisation_complete:      

            return [
                self.eval_growth_rate(self.best_parameters['outside_household_infectivity_scaling'])
                for _ in range(20)
            ]

        else:
            raise Error('Optimisation has not yet been performed. Please run optimise before trying to get fitted samples.')
