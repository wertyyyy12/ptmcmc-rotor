import tensorflow as tf
class config:
  def __init__(self):
    self.dtype=tf.float32
    def tfc(arr):
      return tf.constant(arr, dtype=self.dtype)
    self.seed = 42229
    self.simple_mode = True
    self.output_dim = 4 if not self.simple_mode else 1
    self.num_parameters = 23 if not self.simple_mode else 4
    self.measurement_error = tfc([4., 5., 3., 2.]) if not self.simple_mode else tfc([4.]) # shape = (OUTPUT_DIM,)
    self.number_of_theta_values = 10000
    self.theta_values = tf.cast(tf.reshape(tf.linspace(-5., 5., self.number_of_theta_values), (self.number_of_theta_values,)), self.dtype) # shape = (number_of_theta_values,)
    # print(THETA_VALUES.shape)
    # print(THETA_VALUES)
    self.eager = False # whether to run the tensors in eager mode or not. only for debugging.
    self.strict_prior = False
    if self.eager: 
      tf.config.run_functions_eagerly(True) 
      print("EAGER mode turned on")
    self.true_parameters = tf.cast(tf.linspace(1, 10, self.num_parameters), self.dtype)[tf.newaxis, ...] # shape = (1, NUM_PARAMETERS) print("true_parameters = ", true_parameters)
    self.num_samples_string = "1e4"
    self.num_posterior_samples = int(self.num_samples_string.split("e")[0]) * (10**int(self.num_samples_string.split("e")[1]))
    print("num_posterior_samples = ", num_posterior_samples)
    self.run_identifier = f"{num_samples_string}_strictprior"
    self.num_temperatures = 10 
    self.inverse_temperatures = 0.6**tf.range(self.num_temperatures, dtype=self.dtype)
    self.num_burn_in_steps = 5000 
    self.parameter_labels = [chr(x) for x in range(ord('A'), ord('A') + self.num_parameters)]
    self.step_size = tf.reshape(tf.repeat(tf.constant(.5, dtype=self.dtype), self.num_temperatures * self.num_parameters), (self.num_temperatures, self.num_parameters))
    self.initial_state = tf.zeros(self.num_parameters, dtype=self.dtype)
    # initial_state = tf.constant([10000, -10000], dtype=dtype)
   
