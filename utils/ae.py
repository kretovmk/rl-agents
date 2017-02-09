
import tensorflow as tf
import numpy as np
import keras
import random
import cPickle
import json

from keras.models import model_from_json
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten, UpSampling2D, Reshape, Dropout
from keras.models import Model
from tqdm import tqdm


class AEConvKeras(object):

    def __init__(self, env, inp_shape, dim_hid, n_episodes, max_steps):
        self.env = env
        self.inp_shape = inp_shape
        self.dim_hid = dim_hid
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.states = []
        self.model, self.encoder = self._build_network()

    def collect_data(self):
        for i in tqdm(range(self.n_episodes)):
            state = self.env.reset()
            self.states.append(state)
            terminal = False
            step = 0
            while not terminal and step < self.max_steps:
                action = self.env.action_space.sample()
                state, reward, terminal, _ = self.env.step(action)
                self.states.append(state)
                step += 1

    def project_state(self, preprocessed_state):
        return self.encoder.predict([preprocessed_state])

    def train_model(self, x_train=None, lr=0.001, n_epochs=10, batch_size=256, shuffle=True):
        if x_train is None:
            x_train = np.array(self.states)
        self.model.fit(x_train, x_train, nb_epoch=n_epochs, batch_size=batch_size, shuffle=shuffle)

    def save_model(self):
        self.model.save('autoencoder.h5')
        self.encoder.save('encoder.h5')

    def load_model(self, filepath):
        self.model = keras.models.load_model('autoencoder.h5')
        self.encoder = keras.models.load_model('encoder.h5')

    def _build_network(self):

        inp = Input(shape=self.inp_shape)
        print self.inp_shape

        x = Convolution2D(16, 4, 4, subsample=(2, 2), activation='relu', border_mode='same', dim_ordering='th')(inp)
        x = Convolution2D(16, 4, 4, subsample=(2, 2), activation='relu', border_mode='same', dim_ordering='th')(x)
        x = Flatten()(x)
        encoded = Dense(self.dim_hid, activation='relu')(x)

        x = Dense(8320, activation='relu')(encoded)
        x = Reshape((16, 26, 20), input_shape=(8320,))(x)
        x = UpSampling2D((2, 2), dim_ordering='th')(x)
        x = Convolution2D(16, 4, 4, activation='relu', border_mode='same', dim_ordering='th')(x)
        x = UpSampling2D((2, 2), dim_ordering='th')(x)
        decoded = Convolution2D(12, 4, 4, activation='sigmoid', border_mode='same', dim_ordering='th')(x)

        autoencoder = Model(input=inp, output=decoded)
        encoder = Model(input=inp, output=encoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder, encoder
