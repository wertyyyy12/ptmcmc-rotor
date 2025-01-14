import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class UTILS:
    def cov_mvn(mean, covariance):
        # print(covariance)
        # end params
        # UTILS.print_shape("mean", mean[tf.newaxis, ...])
        # UTILS.print_shape("covariance", covariance)
        scale = tf.linalg.cholesky(covariance)
        # UTILS.print_shape("lin tri", tf.linalg.LinearOperatorLowerTriangular(scale)[tf.newaxis, ...])
        mvn = tfd.MultivariateNormalLinearOperator(
            loc=mean[tf.newaxis, ...],
            scale=tf.linalg.LinearOperatorLowerTriangular(scale)[tf.newaxis, ...],
        )
        return mvn

    def print_shape(name, tensor):
        print("===")
        print("NAME: ", name)
        print(tensor)
        print("SHAPE: ", tensor.shape)
        print("===")
