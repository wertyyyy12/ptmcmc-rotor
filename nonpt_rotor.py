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
# import tensorflow.compat.v2 as tf
from time_logger import start_logger
# tf.enable_v2_behavior()

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
simple_mode = True
OUTPUT_DIM = 4 if not simple_mode else 1
NUM_PARAMETERS = 23 if not simple_mode else 2
MEASUREMENT_ERROR = tfc([4., 5., 3., 2.]) if not simple_mode else tfc([4.]) # shape = (OUTPUT_DIM,)
number_of_theta_values = 20000
# print(tfc([0., 1., .2]).shape)
THETA_VALUES = tf.cast(tf.reshape(tf.linspace(-5., 5., number_of_theta_values), (number_of_theta_values,)), dtype) # shape = (number_of_theta_values,)
# print(THETA_VALUES.shape)
# print(THETA_VALUES)
rng = np.random.default_rng(SEED)
eager = False # whether to run the tensors in eager mode or not. only for debugging.
if eager: 
  tf.config.run_functions_eagerly(True) 
  print("WARNING :::: EAGER mode turned on ::: ")
true_parameters = tf.cast(tf.linspace(1, 9, NUM_PARAMETERS), dtype)[tf.newaxis, ...] # shape = (1, NUM_PARAMETERS) 
print("true_parameters = ", true_parameters)
num_samples_string = "2e3"
num_posterior_samples = int(num_samples_string.split("e")[0]) * (10**int(num_samples_string.split("e")[1]))
print("num_posterior_samples = ", num_posterior_samples)
run_identifier = f"{num_samples_string}_{NUM_PARAMETERS}param"
num_temperatures = 10
inverse_temperatures = 0.6**tf.range(num_temperatures, dtype=dtype)
num_burn_in_steps = num_posterior_samples // 2
assert num_burn_in_steps < num_posterior_samples, "don't burn in more than you sample from the distribution!"
parameter_labels = [chr(x) for x in range(ord('A'), ord('A') + NUM_PARAMETERS)]
step_size = tf.reshape(tf.repeat(tf.constant(.01, dtype=dtype), num_temperatures * NUM_PARAMETERS), (num_temperatures, NUM_PARAMETERS))
# initial_state = tf.ones(NUM_PARAMETERS, dtype=dtype)
initial_state = tf.fill(NUM_PARAMETERS, tf.cast(2., dtype))
target_accept_prob_adaptation = 0.651

use_sigmoid_bijector = False
bijector = tfb.Sigmoid(low=0., high=10.) if use_sigmoid_bijector else tfb.Identity()
# bijector = tfb.Softplus()
# initial_state = tf.constant([10000, -10000], dtype=dtype)

# theta.shape: (batch_theta, theta_size = 1)
# parameters.shape: (batch_parameters, parameters_size = 3)
# shape: (batch_theta, batch_parameters, 4)
# result[:, j, i] gives ith sensor channel (i in {0, 1, 2, 3}) and jth set of parameters (j in {0, 1, ..., 22}) for all theta.
model = None
if simple_mode:
  @tf.function
  def model_(theta, parameters):
    powers = tf.pow(theta[..., tf.newaxis], tf.reshape(tf.range(NUM_PARAMETERS, dtype=dtype), (1, -1)))[:, tf.newaxis, ...]
    intermediate_parameters = tf.cast(tf.transpose(parameters)[tf.newaxis, ...], dtype=dtype)
    result = tf.linalg.matmul(powers, intermediate_parameters)[:, 0, ...]

  #  return tf.cast(tf.stack((result, result * -1., result * -2., result * 2.), axis=-1), dtype=dtype)
    # return tf.cast(tf.stack((result, result * 2.), axis=-1), dtype=dtype)
    return result[..., tf.newaxis]
  model = model_
else:
  @tf.function
  def model_(theta, parameters):
    powers = tf.pow(theta[..., tf.newaxis], tf.reshape(tf.range(NUM_PARAMETERS, dtype=dtype), (1, -1)))[:, tf.newaxis, ...]
    intermediate_parameters = tf.cast(tf.transpose(parameters)[tf.newaxis, ...], dtype=dtype)
    result = tf.linalg.matmul(powers, intermediate_parameters)[:, 0, ...]

    return tf.cast(tf.stack((result, result * -1., result * -2., result * 2.), axis=-1), dtype=dtype)
    # return tf.cast(tf.stack((result, result * 2.), axis=-1), dtype=dtype)
  #  return result[..., tf.newaxis]
  model = model_
 

thetas = THETA_VALUES
print("running to generate ys")
ys = model(thetas, true_parameters)

# @tf.function
# def lnprior(parameters):
# #  print("parameters.shape", parameters.shape) 
# #  print(parameters)
#   # Check if all elements are greater than 0
#   all_greater_than_zero = tf.reduce_all(parameters >= 1, axis=1)
# 
#   # Check if all elements are less than 10
#   all_less_than_ten = tf.reduce_all(parameters <= 10, axis=1)
# 
#   # Check if all conditions are met
#   not_valid_parameters = tf.cast(tf.logical_not(tf.logical_and(all_greater_than_zero, all_less_than_ten)), dtype)
# 
#   neg_inf = tf.constant([-tf.float32.max])
#   # Return 0.0 if valid, -infinity otherwise
# #  print("valid_parameters.shape", valid_parameters.shape)
# #  print(valid_parameters)
# #  print("neg_inf.shape", neg_inf.shape)
# #  print(neg_inf)
# #  print("final result shape", tf.multiply(valid_parameters, neg_inf).shape)
# #  print(tf.multiply(valid_parameters, neg_inf))
# 
#   return tf.multiply(not_valid_parameters, neg_inf)

@tf.function
def lnprior(parameters):
  return tf.constant(0.0, dtype=dtype)

@tf.function
def unnormalized_posterior(parameters):
  likelihoods = tfd.MultivariateNormalDiag(loc=model(thetas, parameters), scale_diag=MEASUREMENT_ERROR)
#  print("likelihoods log prob shape", likelihoods.log_prob(ys).shape)
  return (lnprior(parameters) +
          tf.reduce_sum(likelihoods.log_prob(ys), axis=0)) # axis=0 is IMPORTANT! sum over theta, not over the parameter choices.

# ================ KERNEL CHOICE ================
# def make_kernel_fn(target_log_prob_fn):
#  return tfp.mcmc.HamiltonianMonteCarlo(
#    target_log_prob_fn=target_log_prob_fn,
#    step_size=step_size,
#    num_leapfrog_steps=2)

#def make_kernel_fn(target_log_prob_fn):
# kernel = tfp.mcmc.HamiltonianMonteCarlo(
#   target_log_prob_fn=target_log_prob_fn,
#   step_size=step_size,
#   num_leapfrog_steps=2)
# return tfp.mcmc.SimpleStepSizeAdaptation(
#       kernel,
#       num_adaptation_steps=int(.8 * num_burn_in_steps),
#       target_accept_prob=np.float64(.3))

def make_kernel_fn(target_log_prob_fn):
 kernel = tfp.mcmc.HamiltonianMonteCarlo(
   target_log_prob_fn=target_log_prob_fn,
   step_size=step_size,
   num_leapfrog_steps=3)
 return tfp.mcmc.SimpleStepSizeAdaptation(
       tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=kernel, 
        bijector=bijector
       ),
       num_adaptation_steps=int(.8 * num_burn_in_steps),
       target_accept_prob=target_accept_prob_adaptation)

# def make_kernel_fn(target_log_prob_fn):
#  kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=target_log_prob_fn)
#  return kernel

#   return tfp.mcmc.SimpleStepSizeAdaptation(
#        kernel,
#        num_adaptation_steps=int(.8 * num_posterior_samples),
#        target_accept_prob=np.float64(.6))

# def make_kernel_fn(target_log_prob_fn):
#   kernel = tfp.mcmc.NoUTurnSampler(
#     target_log_prob_fn,
#     step_size,
# )

# ============ end kernel choice ===================

@tf.function
def run_chain(kernel, initial_state, num_posterior_samples=num_posterior_samples, num_burnin_steps=num_burn_in_steps):
  return tfp.mcmc.sample_chain(
    num_results=num_posterior_samples,
    num_burnin_steps=num_burnin_steps,
    current_state=initial_state,
    kernel=kernel,
    trace_fn=lambda current_state, kernel_results: kernel_results
)

remc = tfp.mcmc.ReplicaExchangeMC(
    target_log_prob_fn=unnormalized_posterior,
    inverse_temperatures=inverse_temperatures,
    make_kernel_fn=make_kernel_fn)

non_remc = make_kernel_fn(unnormalized_posterior)

print("running chain")
stop_logger = start_logger("alive", interval=30)
start_time = time.time()
samples, results = run_chain(remc, initial_state)
end_time = time.time()
print(f"chain complete in {end_time - start_time} sec")
print(results)

"""
print("saving to file")
one_string = tf.io.serialize_tensor(samples)
tf.io.write_file("./saved_mcmc/mcmc_saved_chain", one_string)  
print("results were", results)
print(f"saved to file ./saved_mcmc/mcmc_saved_chain")
"""	

try:
    print("plotting corner")
    fig = corner.corner(samples.numpy(),show_titles=True,labels=parameter_labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84], truths=true_parameters.numpy()[0])
    print("corner created")
    fig.savefig(f"{run_identifier}_corner_plot.png", dpi=300, bbox_inches="tight")
    print(f"corner saved to {run_identifier}_corner_plot.png")
    print("samples.shape", samples.numpy().shape)
    for i in range(NUM_PARAMETERS):
      plt.figure()
      print("plotting samples")	
      # plt.xlim(0, 10)  # Replace `xmax` with the desired maximum x-value0 
      plt.plot(samples[0:-1:10, i], c='b', alpha=.3) # IMPORTANT: plots every 100 samples
      print("plotting true value")	
      # print(true_parameters.shape)
      # print(true_parameters)
      plt.hlines(true_parameters[:, i], 0, 1000, zorder=4, color='g', label="$w_{}$".format(i))
      print("success")
      # Add labels, legend, etc. if needed
      plt.legend()
      plt.xlabel('Sample Index')
      plt.ylabel('Parameter Value')

      # Save the plot to a file
      plt.savefig(f'{run_identifier}_trace_plot_{i}.png', dpi=300, bbox_inches='tight')
      plt.close()
    # print("trace.shape", trace.shape)
    # print(trace)
    # Save the figure
except Exception as e:
    print("an error occurred when plotting:")
    print(e) 
    print("the samples were")
    print(samples)

stop_logger()
