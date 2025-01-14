import tensorflow as tf
import tensorflow_probability as tfp

# from bayesian_model import BayesianModel
from bayesian_models.bayesian_model import BayesianModel
from config import CONFIG
from utils import UTILS


class SimpleGaussian(BayesianModel):
    def __init__(self):
        self.input_dim = 0
        self.output_dim = 5
        self.num_parameters = 5
        self.num_data_points = 5000
        self.prior_mean = tf.constant(
            [34, 45, 56, 67, 78], dtype=CONFIG.dtype
        )  # true parameters
        self.data_mean = self.prior_mean * tf.constant(1.3, dtype=CONFIG.dtype)
        _random_matrix = tf.random.uniform(
            shape=(self.output_dim, self.output_dim),
            minval=-10,
            maxval=10,
            dtype=CONFIG.dtype,
        )
        self.data_covariance = tf.matmul(_random_matrix, tf.transpose(_random_matrix))
        self.prior_covariance = self.data_covariance
        self.fake_data = self.generate_data(self.num_data_points)

    def generate_data(self, num_data_points):
        # print(" == generate_data DEBUG == ")
        # print("cov:")
        # print(self.covariance)
        # print("==")
        # print("num_data_points")
        # print(num_data_points)
        generated_data = UTILS.cov_mvn(self.data_mean, self.data_covariance).sample(
            (num_data_points), seed=tfp.random.sanitize_seed(123)
        )
        # print("self.prior_mean")
        # print(self.prior_mean)
        # print("generated_Data")
        # print(generated_data)
        # print(" == end generate_data DEBUG ==")
        return generated_data

    def model(self, data_input, parameters):
        return parameters

    def lnlikelihood(self, data_input, data_output, parameters):
        likelihoods_dist = UTILS.cov_mvn(
            self.model(data_input, parameters), self.data_covariance
        )
        # print("likelihood debug:")
        # print("cov:")
        # print(self.covariance)
        # print("==")
        # print("model:")
        # print(self.model(data_input, parameters))
        # print("==")
        # print("output:")
        # print(data_output)
        # print("==")
        # print("input:")
        # print(data_input)
        # print("END LIKELIHOOD DEBUG")
        return tf.reduce_sum(likelihoods_dist.log_prob(data_output), axis=0)

    def lnprior(self, parameters):
        return UTILS.cov_mvn(self.prior_mean, self.prior_covariance).log_prob(
            parameters
        )

    def _unnormalized_posterior(self, parameters, data_input, data_output):
        likelihoods = self.lnlikelihood(data_input, data_output, parameters)
        # print("debug: ")
        # print(likelihoods)
        # print("==")
        # print(data_input)
        # print("==")
        # print(data_output)
        # print("==")
        # print(parameters)
        return (self.lnprior(parameters) + likelihoods)[0]

    def unnormalized_posterior(self, parameters):
        # fake_data = self.generate_data(self.num_data_points)
        # pdb.set_trace()
        return self._unnormalized_posterior(parameters, self.fake_data, self.fake_data)