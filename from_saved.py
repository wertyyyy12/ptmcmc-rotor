import tensorflow as tf
import tensorflow.compat.v2 as tf
import corner
tf.enable_v2_behavior()

NUM_PARAMETERS = 23
parameter_labels = [chr(x) for x in range(ord('A'), ord('A') + NUM_PARAMETERS)]
dtype=tf.float32
true_parameters = tf.cast(tf.linspace(1, 10, NUM_PARAMETERS), dtype)[tf.newaxis, ...] # shape = (1,     NUM_PARAMETERS)
run_identifier = "2e6"

saved_samples = tf.io.parse_tensor(tf.io.read_file("./saved_mcmc/mcmc_saved_chain"), out_type=dtype)
print("loaded samples: ")
print(saved_samples)

print("making corner")
fig = corner.corner(saved_samples.numpy(),show_titles=True,labels=parameter_labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84], truths=true_parameters.numpy()[0])
print("corner created, saving to file")
fig.savefig(f"{run_identifier}_corner_plot.png", dpi=300, bbox_inches="tight")
print("corner saved")

