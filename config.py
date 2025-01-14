import tensorflow as tf


class CONFIG:
    dtype = tf.float32


class PLOTTING:
    parameter_labels = [chr(x) for x in range(ord("A"), ord("A") + 5)]
    run_identifier = "test"
    plot_thinning_factor = 10
