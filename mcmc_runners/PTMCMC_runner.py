import time
import tensorflow as tf
import tensorflow_probability as tfp
from mcmc_runners.mcmc_runner import MCMC_Runner

class PTMCMC(MCMC_Runner):
    def __init__(self, num_posterior_samples, num_burn_in_steps, initial_state, seed, assumed_measurement_error, actual_measurement_error, num_parameters, target_accept_prob_adapt, inverse_temperatures, make_sub_kernel_fn, eager_mode=False):
        super().__init__(num_posterior_samples, num_burn_in_steps, initial_state, seed, assumed_measurement_error, actual_measurement_error, num_parameters, eager_mode)
        self.target_accept_prob_adapt = target_accept_prob_adapt
        self.inverse_temperatures = inverse_temperatures
        self.make_sub_kernel_fn = make_sub_kernel_fn
        self.is_ptmcmc = True

    def make_kernel_fn(self, target_log_prob_fn):
        return tfp.mcmc.ReplicaExchangeMC(
            target_log_prob_fn=target_log_prob_fn,
            inverse_temperatures=self.inverse_temperatures,
            make_kernel_fn=self.make_sub_kernel_fn
        )
        
        
        
        