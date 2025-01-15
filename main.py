import tensorflow as tf
import numpy as np
import corner
import matplotlib.pyplot as plt
from bayesian_models.simple_gaussian import SimpleGaussian
from mcmc_runners.hmc_runner import HMC
from config import CONFIG, PLOTTING
from utils import UTILS


gaussian_model = SimpleGaussian()
num_params = 5
num_posterior_samples = 100
num_burn_in_steps = num_posterior_samples // 2
hmc_runner = HMC(
    num_posterior_samples=num_posterior_samples,
    num_burn_in_steps=num_burn_in_steps,  #  initial_state=tf.fill([num_params], tf.cast(1000., dtype)),\
    initial_state=tf.fill([num_params], tf.cast(1000.0, CONFIG.dtype)),
    seed=42,
    assumed_measurement_error=tf.constant([]),
    actual_measurement_error=tf.constant([]),
    num_parameters=num_params,
    target_accept_prob_adapt=0.651,
    step_size=0.01,
    num_leapfrog_steps=2,
    eager_mode=False,
)
samples = hmc_runner.mcmc(gaussian_model)


parameter_labels = [
    chr(x) for x in range(ord("A"), ord("A") + gaussian_model.num_parameters)
]
true_parameters = gaussian_model.data_mean
try:
    print("plotting corner")
    fig = corner.corner(
        samples.numpy(),
        show_titles=True,
        labels=parameter_labels,
        plot_datapoints=True,
        quantiles=[0.16, 0.5, 0.84],
        truths=true_parameters,
    )
    print("corner created")
    fig.savefig(
        f"{PLOTTING.run_identifier}_corner_plot.png", dpi=300, bbox_inches="tight"
    )
    print(f"corner saved to {PLOTTING.run_identifier}_corner_plot.png")
    print("samples.shape", samples.numpy().shape)
    for i in range(gaussian_model.num_parameters):
        plt.figure()
        print("plotting samples")
        # plt.xlim(0, 10)  # Replace `xmax` with the desired maximum x-value0
        plt.plot(
            samples[0 : -1 : PLOTTING.plot_thinning_factor, i], c="b", alpha=0.3
        )  # IMPORTANT: plots every `plot_thinning_factor` samples
        print("plotting true value")
        # print(true_parameters.shape)
        # print(true_parameters)
        plt.hlines(
            true_parameters[i],
            0,
            samples.shape[0] // PLOTTING.plot_thinning_factor,
            zorder=4,
            color="g",
            label="$w_{}$".format(i),
        )
        print("success")
        # Add labels, legend, etc. if needed
        plt.legend()
        plt.xlabel("Sample Index")
        plt.ylabel("Parameter Value")

        # Save the plot to a file
        plt.savefig(
            f"{PLOTTING.run_identifier}_trace_plot_{i}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    # print("trace.shape", trace.shape)
    # print(trace)
    # Save the figure
except Exception as e:
    print("an error occurred when plotting:")
    print(e)
    print("the samples were")
    print(samples)
    raise e
finally:
    pass


inverse = tf.linalg.inv
matrixmult = tf.linalg.matmul
inv_prior_cov = inverse(gaussian_model.prior_covariance)
inv_actual_cov = inverse(gaussian_model.data_covariance)
sample_mean = np.mean(samples, axis=0, dtype=np.float32)
UTILS.print_shape("sample mean", sample_mean)
# UTILS.print_shape("inv prior cov", inv_prior_cov)
UTILS.print_shape("prior mean", (gaussian_model.prior_mean)[..., tf.newaxis])
# UTILS.print_shape("inv actual cov", inv_actual_cov
term_A = inverse(inv_prior_cov + (gaussian_model.num_data_points * inv_actual_cov))
# print(matrixmult(inv_prior_cov, CONFIG.prior_mean[..., tf.newaxis]))
# print(matrixmult(inv_actual_cov, sample_mean[..., tf.newaxis
term_B = matrixmult(inv_prior_cov, (gaussian_model.prior_mean)[..., tf.newaxis]) + (gaussian_model.num_data_points * matrixmult(inv_actual_cov, sample_mean[..., tf.newaxis]))

analytical_posterior_mean = matrixmult(term_A, term_B)
analytical_posterior_cov = term_A
print("^^$&@^(*#&$^)")

UTILS.print_shape("sample cov", np.cov(samples, rowvar=False))
print("analytical posterior cov: ")
print(analytical_posterior_cov)
print("data cov: ")
print(gaussian_model.data_covariance)
print("sample mean: ")
print(np.mean(samples, axis=0))
print("analytical posterior mean: ")
print(analytical_posterior_mean)

analytical_posterior_dist = UTILS.cov_mvn(analytical_posterior_mean[:, 0], analytical_posterior_cov)
UTILS.print_shape("mean", analytical_posterior_mean)
UTILS.print_shape("cov", analytical_posterior_cov)
# samples = analytical_posterior_dist.sample(CONFIG.num_posterior_samples)
analytical_posterior_samples = analytical_posterior_dist.sample(hmc_runner.num_posterior_samples)
# analytical_posterior_samples = analytical_posterior_dist.sample(1000)
print(samples.shape)
print(analytical_posterior_samples[:, 0, :].shape)
UTILS.print_shape("analytical posterior samples", analytical_posterior_samples)

fig = corner.corner(analytical_posterior_samples[:, 0, :].numpy(),show_titles=True,labels=PLOTTING.parameter_labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84], truths=true_parameters.numpy(), color="green")

# corner.corner(samples.numpy(),fig=fig, color="red")
plt.savefig(f"{PLOTTING.run_identifier}_corner_plot_analytical.png", dpi=300, bbox_inches="tight")
plt.close()