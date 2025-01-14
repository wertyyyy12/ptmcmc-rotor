class MCMC_Runner:
    def mcmc(self, bayesian_model):
        raise NotImplementedError
        pass

    def __init__(
        self,
        num_posterior_samples,
        num_burn_in_steps,
        initial_state,
        seed,
        assumed_measurement_error,
        actual_measurement_error,
        num_parameters,
        eager_mode=False,
    ):
        self.num_posterior_samples = num_posterior_samples
        self.num_burn_in_steps = num_burn_in_steps
        self.initial_state = initial_state
        self.seed = seed
        self.assumed_measurement_error = assumed_measurement_error
        self.actual_measurement_error = actual_measurement_error
        self.num_parameters = num_parameters
        self.eager_mode = eager_mode
