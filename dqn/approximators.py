
import tensorflow as tf

from base.approximators import QvalueEstimatorBase

class QvalueEstimatorConv(QvalueEstimatorBase):
    """
    Convolutional neural network for Atari games.
    """
    def __init__(self, *args, **kwargs):
        super(QvalueEstimatorConv, self).__init__(*args, **kwargs)

    def _build_network(self):
        conv1 = tf.contrib.layers.conv2d(inputs=self.state_ph, num_outputs=32, kernel_size=(8, 8), stride=(4, 4),
                                         padding='SAME', activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=(4, 4), stride=(2, 2),
                                         padding='SAME', activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=64, kernel_size=(3, 3), stride=(1, 1),
                                         padding='SAME', activation_fn=tf.nn.relu)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        out = tf.contrib.layers.fully_connected(fc1, self.n_actions)
        return out


class QvalueEstimatorDense(QvalueEstimatorBase):
    """
    Dense neural network.
    """

    def __init__(self, *args, **kwargs):
        super(QvalueEstimatorDense, self).__init__(*args, **kwargs)

    def _build_network(self):
        flattened = tf.contrib.layers.flatten(self.state_ph)
        fc1 = tf.contrib.layers.fully_connected(flattened, 16)
        #fc2 = tf.contrib.layers.fully_connected(fc1, 32)
        out = tf.contrib.layers.fully_connected(fc1, self.n_actions)
        return out