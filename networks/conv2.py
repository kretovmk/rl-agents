
import tensorflow as tf
import numpy as np

from networks.base import NetworkBase
from keras.layers import Conv2D, Dense, Input, Flatten, Dropout, BatchNormalization
from keras.models import Model



def build_model(state_size, n_actions,
                conv_filters=(32, 32, 32, 32),
                conv_sizes=(3, 3, 3, 3),
                conv_strides=(2, 2, 2, 2),
                pads=['same']*4,
                conv_droputs=[0.0]*4,
                fc_sizes=(512, 256),
                fc_dropouts=[0.]*2,
                batch_norm=True,
                activation='relu'):

    inputs = Input(shape=state_size)

    conv = inputs
    for (f_num, f_size, f_stride, pad, droput) in zip(conv_filters, conv_sizes, conv_strides, pads, conv_droputs):
        conv = Conv2D(f_num, f_size, f_size,
                      border_mode=pad,
                      subsample=(f_stride, f_stride),
                      activation=activation,
                      dim_ordering='th')(conv)
        if batch_norm:
            conv = BatchNormalization(mode=0, axis=1)(conv)
        if droput > 0:
            conv = Dropout(droput)(conv)


    fc = Flatten()(conv)
    for fc_size, dropout in zip(fc_sizes, fc_dropouts):
        fc = Dense(fc_size, activation=activation)(fc)
        if batch_norm:
            fc = BatchNormalization(mode=1)(fc)
        if droput > 0:
            fc = Dropout(droput)(fc)

    actor = Dense(n_actions, activation='softmax')(fc)

    model = Model(input=inputs, output=actor)

    return model


NET_CONFIGS = {
    'small_cnn': {
        'conv_filters': (32, 32, 32),
        'conv_sizes': (8, 6, 4),
        'conv_strides': (4, 3, 2),
        'pads': ['valid'] * 3,
        'conv_droputs': [0.0] * 3,
        'fc_sizes': (512, ),
        'fc_dropouts': [0.,],
        'batch_norm': True,
        'activation': 'elu'
    }
}



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


