import tensorflow as tf
import tensorflow_probability as tfp

@tf.function
def euclideanPairwiseDistance(x):
    distance = tf.expand_dims(x, 1) - tf.expand_dims(tf.stop_gradient(x), 0)
    return tf.einsum('ijk,kji->ij', distance, tf.transpose(distance))

class RbfKernel:
    @tf.function
    def __call__(self, x):
        normedDist = euclideanPairwiseDistance(x)
        bandwidth = tf.stop_gradient(self.computeBandWidth(normedDist))
        return tf.exp(-0.5 * normedDist / bandwidth**2)

    @tf.function
    def computeBandWidth(self, euclideanPwDistances):
        pwDistanceMedian = tfp.stats.percentile(
            euclideanPwDistances, 50.0, interpolation='midpoint')

        n = tf.constant(euclideanPwDistances.shape[0], dtype = tf.float64)
        return pwDistanceMedian / tf.math.log(n + 1)