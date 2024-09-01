import tensorflow as tf

# theta.shape: (batch_theta, theta_size = 1)
# parameters.shape: (batch_parameters, parameters_size = 3)
# shape: (batch_theta, batch_parameters, 4)
# result[:, j, i] gives ith sensor channel (i in {0, 1, 2, 3}) and jth set of parameters (j in {0, 1, ..., 22}) for all theta.
def get_model(simple_mode):
  model_ret = None
  if simple_mode: # this function is handled weirdly because it is discouraged to use outer variables inside a tf.function
    @tf.function
    def model_(theta, parameters):
      powers = tf.pow(theta[..., tf.newaxis], tf.reshape(tf.range(NUM_PARAMETERS, dtype=dtype), (1, -1)))[:, tf.newaxis, ...]
      intermediate_parameters = tf.cast(tf.transpose(parameters)[tf.newaxis, ...], dtype=dtype)
      result = tf.linalg.matmul(powers, intermediate_parameters)[:, 0, ...]

    #  return tf.cast(tf.stack((result, result * -1., result * -2., result * 2.), axis=-1), dtype=dtype)
      # return tf.cast(tf.stack((result, result * 2.), axis=-1), dtype=dtype)
      return result[..., tf.newaxis]
    model_ret = model_
  else:
    @tf.function
    def model_(theta, parameters):
      powers = tf.pow(theta[..., tf.newaxis], tf.reshape(tf.range(NUM_PARAMETERS, dtype=dtype), (1, -1)))[:, tf.newaxis, ...]
      intermediate_parameters = tf.cast(tf.transpose(parameters)[tf.newaxis, ...], dtype=dtype)
      result = tf.linalg.matmul(powers, intermediate_parameters)[:, 0, ...]

      return tf.cast(tf.stack((result, result * -1., result * -2., result * 2.), axis=-1), dtype=dtype)
      # return tf.cast(tf.stack((result, result * 2.), axis=-1), dtype=dtype)
    #  return result[..., tf.newaxis]
    model_ret = model_
  return model_ret
