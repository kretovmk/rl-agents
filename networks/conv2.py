import tensorflow as tf
import keras
import json
import numpy as np

from tensorflow.python.client import device_lib
from networks.base import NetworkBase
from keras.layers import Conv2D, Dense, Input, Flatten, Dropout, BatchNormalization
from keras.models import Model
from utils.build_model import NET_CONFIGS, build_model



def _build_network(inp_shape, n_actions):
    return build_model(inp_shape, n_actions, **NET_CONFIGS['small_cnn'])



class NetworkCategorialConvKerasPretrained(NetworkBase):

    def __init__(self, scope, inp_shape, n_outputs, fn):
        self.fn = fn
        super(NetworkCategorialConvKerasPretrained, self).__init__(scope, inp_shape, n_outputs)

    def _build_network(self):
        keras.backend.set_learning_phase(0)
        inp = tf.placeholder(shape=(None,) + self.inp_shape, dtype=tf.float32)
        targets = tf.placeholder(shape=(None, self.n_outputs), dtype=tf.float32)
        with tf.variable_scope(self.scope):
            # loading pretrained actor model
            # model = _build_network(self.inp_shape, self.n_outputs)
            # model.load_weights(self.fn)
            model = keras.models.load_model(self.fn)
            # only last layer weights are trained
            self.params = model.trainable_weights[-2:]
            out = model(inp)
            loss = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(out), reduction_indices=[1]))
        return inp, out, targets, loss


class NetworkRegConvKerasPretrained(NetworkBase):

    def __init__(self, scope, inp_shape, n_outputs, fn):
        self.fn = fn
        super(NetworkRegConvKerasPretrained, self).__init__(scope, inp_shape, n_outputs)

    def _build_network(self):
        keras.backend.set_learning_phase(0)
        inp = tf.placeholder(shape=(None,) + self.inp_shape, dtype=tf.float32)
        targets = tf.placeholder(shape=(None, self.n_outputs), dtype=tf.float32)
        with tf.variable_scope(self.scope):
            # loading model pretrained actor model and replacing last layer with single neuron
            # m = _build_network(self.inp_shape, 9)
            # m.load_weights(self.fn)
            m = keras.models.load_model(self.fn)
            # deleting last layer
            inputs = m.layers[0].input
            output = Dense(1, name='Dense_output')(m.layers[-2].output)
            model = Model(input=inputs, output=output)
            # only last layer weights are trained
            self.params = model.trainable_weights[:-2]
            out = model(inp)   
            loss = tf.reduce_mean(tf.squared_difference(out, targets))
        return inp, out, targets, loss