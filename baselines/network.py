
import numpy as np

from baselines.base import BaselineBase


class NetworkBaseline(BaselineBase):

    def __init__(self, approximator=None, optimizer=None):
        super(NetworkBaseline, self).__init__(approximator, optimizer)

    def fit(self, samples, n_epochs=5, batch_size=32):
        x = samples['states']
        y = samples['returns']
        loss = self.approximator.train(x, y, self.optimizer, n_epochs)
        return loss

    def predict(self, samples):
        return np.zeros((len(samples['states']),), dtype=np.float32)