
import tensorflow as tf

from networks.base import NetworkBase
from keras.layers import Input, Dense
from keras.models import Model


class NetworkCategorialDense(NetworkBase):

    def __init__(self, inp_shape, n_outputs, n_hidden):
        self.inp_shape = inp_shape
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        super(NetworkCategorialDense, self).__init__()

    def _build_network(self):
        inp = tf.placeholder(shape=(None,) + self.inp_shape, dtype=tf.float32)
        targets = tf.placeholder(shape=(None, self.n_outputs), dtype=tf.float32)

        # tf contrib model
        out = tf.contrib.layers.flatten(inp)
        for hid in self.n_hidden:
            out = tf.contrib.layers.fully_connected(out, hid,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer())
            out = tf.nn.relu(out)
        logits = tf.contrib.layers.fully_connected(out, self.n_outputs,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer())

        out = tf.nn.softmax(logits)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits, targets)
        return inp, out, targets, loss


class NetworkCategorialDenseKeras(NetworkBase):

    def __init__(self, inp_shape, n_outputs, n_hidden):
        self.inp_shape = inp_shape
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        super(NetworkCategorialDenseKeras, self).__init__()

    def _build_network(self):
        inp = tf.placeholder(shape=(None,) + self.inp_shape, dtype=tf.float32)
        targets = tf.placeholder(shape=(None, self.n_outputs), dtype=tf.float32)

        # keras model
        inputs = Input(shape=self.inp_shape)
        out = inputs
        for i, hid in enumerate(self.n_hidden):
            out = Dense(output_dim=hid, activation='relu', )(out)
        out = Dense(output_dim=self.n_outputs)(out)
        model = Model(input=inputs, output=out)
        logits = model(inp)

        out = tf.nn.softmax(logits)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits, targets)
        return inp, out, targets, loss

