import tensorflow as tf
import corner
import matplotlib.pyplot as plt
from bayesian_models.simple_gaussian import SimpleGaussian
from mcmc_runners.hmc_runner import HMC
from config import CONFIG, PLOTTING


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
