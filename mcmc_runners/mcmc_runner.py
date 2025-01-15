import time
import tensorflow as tf
import tensorflow_probability as tfp

class MCMC_Runner:
    # def mcmc(self, bayesian_model):
    #     raise NotImplementedError

    def __init__(
        self,
        num_posterior_samples,
        num_burn_in_steps,
        initial_state,
        seed,
        assumed_measurement_error,
        actual_measurement_error,
        num_parameters,
        is_ptmcmc=False,
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
        self.is_ptmcmc = is_ptmcmc

    @tf.function
    def run_chain(
        self,
        bayesian_model,
    ):
        lnposterior = lambda parameters: bayesian_model.unnormalized_posterior(parameters, is_ptmcmc=self.is_ptmcmc)
        kernel = self.make_kernel_fn(lnposterior)
        tf.print("running chain")
        start_time = time.time()
        samples, results = tfp.mcmc.sample_chain(
            num_results=self.num_posterior_samples,
            num_burnin_steps=self.num_burn_in_steps,
            current_state=self.initial_state,
            kernel=kernel,
            trace_fn=lambda current_state, kernel_results: kernel_results,
        )
        end_time = time.time()
        tf.print(f"chain complete in {end_time - start_time} sec")
        return samples
    def make_kernel_fn(self, target_log_prob_fn):
        raise NotImplementedError
