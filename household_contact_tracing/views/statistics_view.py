# Code to estimate the growth rate of a simulated branching process
from household_contact_tracing.views.branching_process_view import BranchingProcessView
from household_contact_tracing.branching_process_model import BranchingProcessModel
from household_contact_tracing.branching_process_state import MaxNodesInfectiousState, ReadyState, RunningState, ExtinctState
from household_contact_tracing.exceptions import Error, ModelStateError
import scipy.stats as ss
import statsmodels.api as sm
import numpy as np

class StatisticsView(BranchingProcessView):

    def __init__(self, model: BranchingProcessModel):
        """View that performs statistical analysis of a simulated epidemic.
        For example, by estimating the growth rate of an epidemic, or estimating the household secondary attack rate.
        
        Args:
            model (BranchingProcessModel): Input branching process epidemic to be analysed.
        """
        self._model = model
        self.show = False

    def model_state_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in model state (e.g. running, extinct, timed-out)

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        # nothing to do here, usually it only makes sense to estimate the growth rate after the simulation is complete.
        pass

    def model_step_increment(self, subject: BranchingProcessModel):
        """
        Respond to single step increment in simulation

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        # nothing to do here, usually it only makes sense to estimate the growth rate after the simulation is complete.
        pass

    def model_simulation_stopped(self, subject: BranchingProcessModel):
        """
        Respond to end of simulation run

            Parameters:
                subject (BranchingProcessModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        if self.show:
            self._estimate_growth_rate()

    def graph_change(self, subject: BranchingProcessModel):
        """
        Respond to changes in graph (nodes/households network)

            Parameters:
                subject (SimulationModel): The branching process model being displayed by this simulation view.

            Returns:
                None
        """
        # nothing to do here, usually it only makes sense to estimate the growth rate after the simulation is complete.
        pass

    def set_display(self, show: bool):
        """
        Sets whether this view is displayed or not.

            Parameters:
                show (bool): To display this view, set to True

            Returns:
                None
        """
        self.show = show

    def get_infection_times(self):
        """
        Returns a list containing the times at which each node was infected
        """
        return [node.time_infected for node in self._model.network.all_nodes()]

    def get_daily_incidence(self):
        """Returns a list of the new infections at each time point.

        The list contains [time, incidence] pairs
        """
        
        infection_times = self.get_infection_times()
        return([
            infection_times.count(t)
            for t in range(self._model.time)
        ])

    def _estimate_growth_rate(self, discard_first_n_days: int = 10, verbose = True):
        """Uses Poisson regression to estimate the growth rate of the epidemic.
        The first few days of a simulation are typically discarded while the process becomes mixed
        after it's artificial initial conditions

        Args:
            discard_first_n_days (int, optional): estimate growth rate from data after the first n days. Defaults to 10.
        """

        if isinstance(self._model.state, ReadyState):
            raise ModelStateError(self._model.state, 'Simulation has not started yet. Cannot estimate growth rate.')

        # we work out how many 
        time = self._model.time
        num_eligible_dates = time - discard_first_n_days

        if num_eligible_dates < 2:
            # there is not enough data to estimate the growth rate

            if isinstance(self._model.state, RunningState):
                raise Error("""Cannot estimate growth rate due to insufficient eligible dates.
                This simulation is still running, consider continuing the simulation before estimating the growth rate.""")

            elif isinstance(self._model.state, ExtinctState):
                raise Error("""Cannot estimate growth rate due to insufficient eligible dates.
                This simulation went extinct, possibly before discard_first_n_days. Consider starting the simulation with more infections""")

            elif isinstance(self._model.state, MaxNodesInfectiousState):
                raise Error("""Cannot estimate growth rate due to insufficient eligible dates.
                This simulation exceeded the maximum number of infectious nodes. Consider raising max_active_infections when simulating.""")

            else:
                raise Error("""Cannot estimate growth rate due to insufficient eligible dates.""")

        else:
            # there is some data that can be used to estimate the growth rate. Perform analysis

            if verbose:
                print(f'Estimating growth rate using {num_eligible_dates} time periods')

            # the incidence after the first n days
            y = self.get_daily_incidence()[discard_first_n_days:]

            # create a simple design matrix
            X = [[t] for t in range(num_eligible_dates)]
            X = sm.add_constant(X, prepend=False)
            
            glm_poisson = sm.GLM(y, X, family=sm.families.Poisson())
            self.glm_poisson = glm_poisson.fit()

    def get_growth_rate(self, discard_first_n_days: int = 10, verbose: bool = True):
        """Returns the growth rate of the simulated epidemic, estimated using poisson regression.

        The first few days of a simulation are typically discarded while the process becomes mixed
        after it's artificial initial conditions.

        Args:
            discard_first_n_days (int, optional): estimate growth rate from data after the first n days. Defaults to 10.
        """
        self._estimate_growth_rate(discard_first_n_days, verbose)
        
        return self.glm_poisson.params[0]

    def growth_rate_summary(self, discard_first_n_days: int = 10, alpha: float = 0.05, glm_summary: bool = False):
        """Returns the growth rate of the simulated epidemic, estimated using poisson regression.

        The first few days of a simulation are typically discarded while the process becomes mixed
        after it's artificial initial conditions.

        Args:
            discard_first_n_days (int, optional): estimate growth rate from data after the first n days. Defaults to 10.
        """
        
        self._estimate_growth_rate(discard_first_n_days, verbose = False)

        num_eligible_dates  = self._model.time - discard_first_n_days
        growth_rate         = self.glm_poisson.params[0]
        growth_rate_ci      = self.glm_poisson.conf_int(alpha = alpha, cols = [0])[0]
        doubling_time       = np.log(2) / np.log(1 + growth_rate)
        doubling_time_ci    = np.log(2) / np.log(1 + np.array(growth_rate_ci))

        if glm_summary:
            print(self.glm_poisson.summary())
        print(f'{num_eligible_dates} time periods were used to estimate the growth rate.')
        print(f'The estimated growth rate was {round(growth_rate*100, 2)}% ({100*(1-alpha)}% CI: {round(growth_rate_ci[0]*100,2)}-{round(growth_rate_ci[1]*100,2)}%) per day.')
        print(f'The estimated doubling time is {round(doubling_time, 2)} ({100*(1-alpha)}% CI: {round(doubling_time_ci[1],2)}-{round(doubling_time_ci[0],2)}) days.')

    def _estimate_household_secondary_attack_rate(self, use_first_generation_only: bool = False) -> None:
        if isinstance(self._model.state, ReadyState):
            raise ModelStateError(self._model.state, 'Simulation has not started yet. Cannot estimate growth rate.')

        if use_first_generation_only:
            households_with_completed_local_epidemics = [
                household 
                for household 
                in self._model.network.all_households
                if household.local_epidemic_completed
                and household.id in self._model.infection.starting_households
            ]
        else:
            households_with_completed_local_epidemics = [
                household 
                for household 
                in self._model.network.all_households
                if household.local_epidemic_completed
            ]

        # size of household - number of remaining susceptibles = final size.
        # we subtract 1, to work out the number of non-index secondary infections
        self.total_infected = sum([
            household.size - household.susceptibles - 1 
            for household 
            in households_with_completed_local_epidemics
        ])

        # we subtract 1, to work out the number of non-index exposed individuals
        self.total_exposed = sum([
            household.size - 1 for household in households_with_completed_local_epidemics 
        ])

        self.n_households_with_completed_local_epidemics = len(households_with_completed_local_epidemics)

        self.household_sar = self.total_infected / self.total_exposed
        
        # calculating some confidence intervals using the good ol' Jefferys interval
        self.household_sar_ci = ss.beta.interval(alpha = 0.95, a = self.total_infected + 0.5, b = self.total_exposed - self.total_infected + 0.5)

    def get_hh_sar(self):
        self._estimate_household_secondary_attack_rate(use_first_generation_only=True)
        return self.household_sar

    def household_secondary_attack_rate_summary(self, use_first_generation_only: bool = False, alpha: float = 0.95) -> None:
        """Estimates the household secondary attack rate, and prints and interpretable output.

        Args:
            use_first_generation_only (bool, optional): Use the first generation of the household epidemic only to estimate the household secondary attack. Defaults to False.
        """
        self._estimate_household_secondary_attack_rate(use_first_generation_only)


        print('Household secondary attack rate summary:')
        print(f'{self.n_households_with_completed_local_epidemics} local household epidemics were eligible to be included.')
        if use_first_generation_only:
            print('Only the first generation of the household epidemic was included in this calculation.')
        else:
            print('All households with completed local epidemics were included. This may lead to a biased sample, as it is possible that local epidemics with a long duration were not included.')
        print(f'There were {self.total_exposed} non-index susceptible individuals exposed, of which {self.total_infected} were infected.')
        print(f'This yields a household secondary attack rate of {round(self.household_sar*100)}% ({int(alpha * 100)}% CI: {round(self.household_sar_ci[0]*100)}-{round(self.household_sar_ci[1]*100)}%).')
