import time
import tensorflow as tf
import tensorflow_probability as tfp
from mcmc_runners.mcmc_runner import MCMC_Runner


class NUTS(MCMC_Runner):
    def __init__(
        self,
        num_posterior_samples,
        num_burn_in_steps,
        initial_state,
        seed,
        assumed_measurement_error,
        actual_measurement_error,
        num_parameters,
        target_accept_prob_adapt,
        step_size,
        num_leapfrog_steps,
        eager_mode=False,
    ):
        super().__init__(
            num_posterior_samples,
            num_burn_in_steps,
            initial_state,
            seed,
            assumed_measurement_error,
            actual_measurement_error,
            num_parameters,
            is_ptmcmc=False,
            eager_mode=eager_mode,
        )
        self.target_accept_prob_adapt = target_accept_prob_adapt
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.num_adaptation_steps = int(num_burn_in_steps * 0.8)

    def make_kernel_fn(self, target_log_prob_fn):

        # print("start debug: ")
        # print(self.step_size)
        # print(self.num_leapfrog_steps)
        # print(self.target_accept_prob_adapt)
        # print("end debug: ")
        kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn, step_size=self.step_size
        )
        return tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=int(self.num_adaptation_steps),
            target_accept_prob=self.target_accept_prob_adapt,
        )

    # @tf.function
    # def mcmc(self, bayesian_model):
    #     # curry the unnormalized posterior function with the is_ptmcmc flag set to False
    #     lnposterior = lambda parameters: bayesian_model.unnormalized_posterior(parameters, is_ptmcmc=False)


    #     # def make_kernel_fn(target_log_prob_fn):
    #     #  return tfp.mcmc.HamiltonianMonteCarlo(
    #     #    target_log_prob_fn=target_log_prob_fn,
    #     #    step_size=CONFIG.step_size,
    #     #    num_leapfrog_steps=CONFIG.num_leapfrog_steps)

    #     # def make_kernel_fn(target_log_prob_fn):
    #     #  kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=target_log_prob_fn)
    #     #  return kernel

    #     #   return tfp.mcmc.SimpleStepSizeAdaptation(
    #     #        kernel,
    #     #        num_adaptation_steps=int(.8 * num_posterior_samples),
    #     #        target_accept_prob=np.float64(.6))

    #     # def make_kernel_fn(target_log_prob_fn):
    #     #   kernel = tfp.mcmc.NoUTurnSampler(
    #     #     target_log_prob_fn,
    #     #     step_size,
    #     # )

    #     # ============ end kernel choice ===================


    #     # remc = tfp.mcmc.ReplicaExchangeMC(
    #     #     target_log_prob_fn=lnposterior,
    #     #     inverse_temperatures=CONFIG.inverse_temperatures,
    #     #     make_kernel_fn=make_kernel_fn)

    #     hmc = self.make_kernel_fn(lnposterior)
    #     print("bootstrap results...")
    #     print(hmc.bootstrap_results(self.initial_state))
    #     tf.print("running chain")
    #     start_time = time.time()
    #     samples, results = self.run_chain(hmc)
    #     end_time = time.time()
    #     tf.print(f"chain complete in {end_time - start_time} sec")
    #     return samples
