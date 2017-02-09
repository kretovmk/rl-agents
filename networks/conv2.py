import tensorflow as tf
import keras
import numpy as np

from networks.base import NetworkBase
from keras.layers import Conv2D, Dense, Input, Flatten, Dropout, BatchNormalization
from keras.models import Model


class NetworkCategorialConvKerasPretrained(NetworkBase):

    def __init__(self, scope, inp_shape, n_outputs, fn):
        self.fn = fn
        super(NetworkCategorialConvKerasPretrained, self).__init__(scope, inp_shape, n_outputs)

    def _build_network(self):
        inp = tf.placeholder(shape=(None,) + self.inp_shape, dtype=tf.float32)
        with tf.variable_scope(self.scope):
            # loading pretrained actor model
            model = keras.models.load_model(self.fn)
            # only last layer weights are trained
            self.params = model.weights[-2:]
            out = model(inp)
            targets, loss = None, None  # not needed for policy
        return inp, out, targets, loss


class NetworkRegConvKerasPretrained(NetworkBase):

    def __init__(self, scope, inp_shape, n_outputs, fn):
        self.fn = fn
        super(NetworkRegConvKerasPretrained, self).__init__(scope, inp_shape, n_outputs)

    def _build_network(self):
        inp = tf.placeholder(shape=(None,) + self.inp_shape, dtype=tf.float32)
        targets = tf.placeholder(shape=(None, self.n_outputs), dtype=tf.float32)
        with tf.variable_scope(self.scope):
            # loading model pretrained actor model and replacing last layer with single neuron
            m = keras.models.load_model(self.fn)
            inputs = m.layers[0].input
            output = Dense(1)(m.layers[-2].output)
            model = Model(input=inputs, output=output)
            # only last layer weights are trained
            self.params = model.weights[:-2]
            out = model(inp)   
            loss = tf.reduce_mean(tf.squared_difference(out, targets))
        return inp, out, targets, loss