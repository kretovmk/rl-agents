
import tensorflow as tf

from base.approximators import PolicyBase, ValueEstimatorBase


class PolicyDense(PolicyBase):

    def __init__(self, *args, **kwargs):
        super(PolicyDense, self).__init__(*args, **kwargs)

    def _build_network(self):
        out = tf.contrib.layers.flatten(self.state_ph)
        out = tf.contrib.layers.fully_connected(out, 16)
        out = tf.contrib.layers.fully_connected(out, self.n_actions)
        out = tf.nn.softmax(out)
        return out


class ValueDense(ValueEstimatorBase):

    def __init__(self, *args, **kwargs):
        super(ValueDense, self).__init__(*args, **kwargs)

    def _build_network(self):
        out = tf.contrib.layers.flatten(self.state_ph)
        out = tf.contrib.layers.fully_connected(out, 16)
        out = tf.contrib.layers.fully_connected(out, 1)
        return out

