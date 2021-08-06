"""
Code that deals with common hyperparameter optimisation routines.

Often, we want to calibrate an epidemic to a desired growth rate and household secondary attack rate.
"""

from abc import ABC
from household_contact_tracing.branching_process_models import HouseholdLevelTracing
from household_contact_tracing.branching_process_controller import BranchingProcessController
from ax import optimize
from copy import Error, copy
import math

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
            desired_hh_sar: float,
            asymptomatic_prob: float,
            asymptomatic_relative_infectivity: float,
            infection_reporting_prob: float,
            reduce_contacts_by: float,
            starting_infections: int = 100,
            starting_infections_hh_sar: int = 1000
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
        self.desired_hh_sar      = desired_hh_sar
        self.starting_infections_hh_sar = starting_infections_hh_sar

        self.optimisation_complete = False

    def eval_metrics(
            self,
            household_pairwise_survival_prob: float,
            outside_household_infectivity_scaling: float,
            state_criteria: dict = {},
            verbose: bool = True) -> float:
        """Sets up a model, runs it, and returns the evaluated growth rate.

        Args:
            outside_household_infectivity_scaling (float): controls how infectious global contacts are
            max_time (int, optional): upper limit of days to simulate. Defaults to 20.
            max_active_infections (int, optional): simulation ends early if maximum number of infections is exceeded. Defaults to 1e5.
        """

        params = copy(self.fixed_params)
        params['outside_household_infectivity_scaling'] = outside_household_infectivity_scaling
        params['household_pairwise_survival_prob']      = household_pairwise_survival_prob
        
        # run a simulation to get the growth rate of the epidemic
        controller = BranchingProcessController(HouseholdLevelTracing(params))
        controller.csv_view.set_display(False)
        controller.run_simulation(state_criteria)

        # use a different simulation method to get the household secondary attack rate of the epidemic
        params['starting_infections'] = self.starting_infections_hh_sar # use a higher number of starting infections
        controller_hh_sar = BranchingProcessController(HouseholdLevelTracing(params))
        controller_hh_sar.csv_view.set_display(False)
        controller_hh_sar.run_hh_sar_simulation(
            state_criteria = {
                'infection_threshold': math.inf,
                'max_time': math.inf
                }
        )

        return {
            'growth_rate': controller.statistics_view.get_growth_rate(verbose = verbose),
            'hh_sar': controller_hh_sar.statistics_view.get_hh_sar()
        }


    def evaluate_fit(
            self,
            household_pairwise_survival_prob,
            outside_household_infectivity_scaling,
            verbose: bool = True,
            state_criteria: dict = {}) -> float:
        
        metrics = self.eval_metrics(
            household_pairwise_survival_prob,
            outside_household_infectivity_scaling,
            state_criteria,
            verbose
        )

        return abs(self.desired_growth_rate - metrics['growth_rate']) + abs(self.desired_hh_sar - metrics['hh_sar'])

    def optimise(self, 
            outside_household_infectivity_scaling_range: list[float],
            household_pairwise_survival_prob_range: list[float],
            total_trials: int = 20,
            state_criteria: dict = {}):
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
                    "value_type": "float"
                },
                {
                    "name": "household_pairwise_survival_prob",
                    "type": "range",
                    "bounds": household_pairwise_survival_prob_range,
                    "value_type": "float"
                }
            ],
            evaluation_function = lambda pars: self.evaluate_fit(
                household_pairwise_survival_prob = pars["household_pairwise_survival_prob"],
                outside_household_infectivity_scaling = pars["outside_household_infectivity_scaling"],
                verbose = False,
                state_criteria = state_criteria
                ),
            minimize            = True,
            total_trials        = total_trials
        )
        
        self.optimisation_complete  = True

        return self.best_parameters, self.values

    def get_fitted_model_metric_samples(
        self,
        n_obs: int = 10,
        state_criteria: dict = {}) -> list[dict]:
        """If optimisation has been completed, this method generates sample of the growth rate using
        the results from the optimisation step.


        Args:
            n_obs (int, optional): Number of fitted sample of the growth rate to get. Defaults to 10.

        Returns:
            [list]: A list containing fitted samples of the growth rate.
        """

        if self.optimisation_complete:      

            return [
                self.eval_metrics(
                    household_pairwise_survival_prob = self.best_parameters['household_pairwise_survival_prob'],
                    outside_household_infectivity_scaling = self.best_parameters['outside_household_infectivity_scaling'],
                    state_criteria = state_criteria,
                    verbose = False)
                for _ in range(n_obs)
            ]

        else:
            raise Error('Optimisation has not yet been performed. Please run optimise before trying to get fitted samples.')
