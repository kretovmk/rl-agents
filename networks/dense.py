

import tensorflow as tf

from networks.base import NetworkBase


class NetworkCategorialDense(NetworkBase):

    def __init__(self, inp_shape, n_outputs, n_hidden):
        self.inp_shape = inp_shape
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        super(NetworkCategorialDense, self).__init__()

    def _build_network(self):
