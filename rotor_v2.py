# Commented out IPython magic to ensure Python compatibility.
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import threading
import pickle
import time
# import seaborn as sns
import logging
import sys
# !pip install -U corner
import corner
# !pip install -U emcee
# import emcee
import tensorflow as tf
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp

# sns.reset_defaults()
# sns.set_context(context='talk',font_scale=0.7)
plt.rcParams['image.cmap'] = 'viridis'

# %matplotlib inline

tfd = tfp.distributions
tfb = tfp.bijectors
from tensorflow_probability.python.mcmc import kernel as kernel_base
# from tf.math import polyval

def npf(array_):
  return np.array([float(x) for x in array_])

dtype=tf.float32
def tfc(arr):
  return tf.constant(arr, dtype=dtype)


SEED = 42229
OUTPUT_DIM = 1
NUM_PARAMETERS = 5
MEASUREMENT_ERROR = tfc([4.,]) # shape = (OUTPUT_DIM,)
THETA_VALUES = tfc([1., 2., 3.]) # shape = (number_of_theta_values,)
rng = np.random.default_rng(SEED)

true_parameters = tf.cast(tf.linspace(1, 10, NUM_PARAMETERS), dtype)[tf.newaxis, ...] # shape = (1, NUM_PARAMETERS)
print("true_parameters = ", true_parameters)
run_identifier = "SPLIT"
num_temperatures = 7
inverse_temperatures = 0.6**tf.range(num_temperatures, dtype=dtype)
num_posterior_samples = 30000
num_burn_in_steps = 100
parameter_labels = [chr(x) for x in range(ord('A'), ord('A') + NUM_PARAMETERS)]
step_size = tf.reshape(tf.repeat(tf.constant(.1, dtype=dtype), num_temperatures * NUM_PARAMETERS), (num_temperatures, NUM_PARAMETERS))
initial_state = tf.zeros(NUM_PARAMETERS, dtype=dtype)
# initial_state = tf.constant([10000, -10000], dtype=dtype)

# theta.shape: (batch_theta, theta_size = 1)
# parameters.shape: (batch_parameters, parameters_size = 3)
# shape: (batch_theta, batch_parameters, 4)
# result[:, j, i] gives ith sensor channel (i in {0, 1, 2, 3}) and jth set of parameters (j in {0, 1, ..., 22}) for all theta.
@tf.function
def model(theta, parameters):
  powers = tf.pow(theta[..., tf.newaxis], tf.reshape(tf.range(NUM_PARAMETERS, dtype=dtype), (1, -1)))[:, tf.newaxis, ...]
  print(powers.shape)
  intermediate_parameters = tf.cast(tf.transpose(parameters)[tf.newaxis, ...], dtype=dtype)
  print(intermediate_parameters.shape)
  result = tf.linalg.matmul(powers, intermediate_parameters)[:, 0, ...]

  # return tf.cast(tf.stack((result, result * 2., result * 3., result * 4.), axis=-1), dtype=dtype)
  # return tf.cast(tf.stack((result, result * 2.), axis=-1), dtype=dtype)
  return result[..., tf.newaxis]

thetas = THETA_VALUES
ys = model(thetas, true_parameters)

@tf.function
def lnprior(parameters):
  return tfc(0.0)

@tf.function
def unnormalized_posterior(parameters):
  # print("predicted values shape", model(thetas, parameters).shape)
  print("model(thetas, parameters).shape", model(thetas, parameters).shape)
  print("parameters.shape", parameters.shape)
  print("MEASUREMENT_ERROR.shape", MEASUREMENT_ERROR.shape)
  # print(model(thetas, parameters)[1, 0, 0])
  # print(model(thetas, parameters)[:, 1, 0])
  # print(model(thetas, parameters)[:, 0, 1])
  likelihoods = tfd.MultivariateNormalDiag(loc=model(thetas, parameters), scale_diag=MEASUREMENT_ERROR)
  print("likelihoods log prob shape", likelihoods.log_prob(ys).shape)
  # print("reduced sum, with specified axis shape", tf.reduce_sum(likelihoods.log_prob(ys), axis=0).shape)
  return (lnprior(parameters) +
          tf.reduce_sum(likelihoods.log_prob(ys), axis=0)) # axis=0 is IMPORTANT! sum over theta, not over the parameter choices.

# print(tf.concat((true_parameters, true_parameters + 1.), axis=0))
print(unnormalized_posterior(tf.concat((true_parameters, true_parameters + 1., true_parameters + 2., true_parameters + 3.), axis=0)))

# ================ KERNEL CHOICE ================
# def make_kernel_fn(target_log_prob_fn):
#  return tfp.mcmc.HamiltonianMonteCarlo(
#    target_log_prob_fn=target_log_prob_fn,
#    step_size=step_size,
#    num_leapfrog_steps=2)

def make_kernel_fn(target_log_prob_fn):
  kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    step_size=step_size,
    num_leapfrog_steps=2)
  return tfp.mcmc.SimpleStepSizeAdaptation(
        kernel,
        num_adaptation_steps=int(.8 * num_burn_in_steps),
        target_accept_prob=np.float64(.3))


# ============ end kernel choice ===================

@tf.function
def run_chain(kernel, initial_state, num_posterior_samples=num_posterior_samples, num_burnin_steps=num_burn_in_steps):
  return tfp.mcmc.sample_chain(
    num_results=num_posterior_samples,
    num_burnin_steps=num_burnin_steps,
    current_state=initial_state,
    kernel=kernel,
#    trace_fn=lambda current_state, kernel_results: kernel_results
    trace_fn=partial(trace_fn, summary_freq=5)
)

@tf.function
def run_chain_block(kernel, initial_state, num_posterior_samples=num_posterior_samples, num_burnin_steps=num_burn_in_steps):
  print(kernel)
  current_state = initial_state
  kernel_results = kernel.bootstrap_results(current_state)
  chain_blocks = []
  trace_blocks = []
  # current_state = initial_state
  num_blocks = 3
  for i in range(num_blocks):
    print(f"running block {i}")
    chain, trace, kernel_results = tfp.mcmc.sample_chain(
      num_posterior_samples,
      num_burnin_steps=num_burnin_steps,
      current_state=current_state,
      kernel=kernel,
      trace_fn=lambda current_state, kernel_results: kernel_results,
      previous_kernel_results=kernel_results,
      return_final_kernel_results=True,
    )
    print("chain ended at ", chain[-1, :])
    print(tf.nest.map_structure(lambda x: x[-1], chain)[0])
    current_state = chain[-1, :]
    chain_blocks.append(chain)
    trace_blocks.append(trace)

    with open("mcmc_inter_kernel", "wb") as f:
      pickle.dump(kernel_results, f)
    one_string = tf.strings.format("{}", (chain))
    print("writing file")
    tf.io.write_file("mcmc_intermediate_results", one_string)  
    print("written?")
    # print(chain[-1, :])
    print(trace)
    # print(kernel_results)

	# Do your partial analysis here.


  print("what's happening")
  full_chain = tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=0), *chain_blocks)
  print("something is happening")
  full_trace = tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=0), *trace_blocks)
  print("finally..?")
  return full_chain, full_trace
  # return full_chain, full_trace


remc = tfp.mcmc.ReplicaExchangeMC(
    target_log_prob_fn=unnormalized_posterior,
    inverse_temperatures=inverse_temperatures,
    make_kernel_fn=make_kernel_fn)

samples, trace = run_chain_block(remc, initial_state)
# samples, kernel_results = run_chain(remc, initial_state)


try:
    fig = corner.corner(samples.numpy(),show_titles=True,labels=parameter_labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84], truths=true_parameters.numpy()[0])
    fig.savefig(f"{run_identifier}_corner_plot.png", dpi=300, bbox_inches="tight")
    print("samples.shape", samples.numpy().shape)
    for i in range(NUM_PARAMETERS):
      plt.figure()
      print("plotting samples")	
      # plt.xlim(0, 10)  # Replace `xmax` with the desired maximum x-value0 
      plt.plot(samples[0:-1:100, i], c='b', alpha=.3)
      print("ployying")	
      print(true_parameters.shape)
      print(true_parameters)
      plt.hlines(true_parameters[:, i], 0, 1000, zorder=4, color='g', label="$w_{}$".format(i))
      print("success")
      # Add labels, legend, etc. if needed
      plt.legend()
      plt.xlabel('Sample Index')
      plt.ylabel('Value')

      # Save the plot to a file
      plt.savefig(f'{run_identifier}_trace_plot_{i}.png', dpi=300, bbox_inches='tight')
    # print("trace.shape", trace.shape)
    # print(trace)
    # Save the figure
except Exception as e:
    print("an error occurred when plotting:")
    print(e) 
    raise e

