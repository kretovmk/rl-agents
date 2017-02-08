
import tensorflow as tf
import numpy as np

from networks.base import NetworkBase
from keras.layers import Input, Dense, Convolution2D, Flatten
from keras.models import Model



class NetworkCategorialConvKeras(NetworkBase):

    def __init__(self, scope, inp_shape, n_outputs):
        super(NetworkCategorialConvKeras, self).__init__(scope, inp_shape, n_outputs)

    def _build_network(self):
        inp = tf.placeholder(shape=(None,) + self.inp_shape, dtype=tf.float32)
        targets = tf.placeholder(shape=(None, self.n_outputs), dtype=tf.float32)

        # keras model
        with tf.variable_scope(self.scope):
            inputs = Input(shape=self.inp_shape)
            out = inputs
            out = Convolution2D(16, 4, 4, subsample=(2, 2), activation='relu', border_mode='same', dim_ordering='th')(out)
            out = Convolution2D(16, 4, 4, subsample=(2, 2), activation='relu', border_mode='same', dim_ordering='th')(out)
            out = Flatten()(out)
            out = Dense(64, activation='relu')(out)
            out = Dense(output_dim=self.n_outputs)(out)
            model = Model(input=inputs, output=out)
            logits = model(inp)
            out = tf.nn.softmax(logits)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
            self.inp, self.out, self.targets, self.loss = inp, out, targets, loss
            self.params = model.trainable_weights
        return inp, out, targets, loss


class NetworkRegConvKeras(NetworkBase):

    def __init__(self, scope, inp_shape, n_outputs):
        super(NetworkRegConvKeras, self).__init__(scope, inp_shape, n_outputs)

    def _build_network(self):
        inp = tf.placeholder(shape=(None,) + self.inp_shape, dtype=tf.float32)
        targets = tf.placeholder(shape=(None, self.n_outputs), dtype=tf.float32)

        # keras model
        with tf.variable_scope(self.scope):
            inputs = Input(shape=self.inp_shape)
            out = inputs
            out = Convolution2D(16, 4, 4, subsample=(2, 2), activation='relu', border_mode='same', dim_ordering='th')(out)
            out = Convolution2D(16, 4, 4, subsample=(2, 2), activation='relu', border_mode='same', dim_ordering='th')(out)
            out = Flatten()(out)
            out = Dense(64, activation='relu')(out)
            out = Dense(output_dim=self.n_outputs)(out)
            model = Model(input=inputs, output=out)
            out = model(inp)
            loss = tf.reduce_mean(tf.squared_difference(out, targets))
            self.inp, self.out, self.targets, self.loss = inp, out, targets, loss
            self.params = model.trainable_weights

        return inp, out, targets, loss


