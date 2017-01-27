
import numpy as np

from baselines.base import BaselineBase


class ZeroBaseline(BaselineBase):

    def __init__(self, approximator=None, optimizer=None):
        super(ZeroBaseline, self).__init__(approximator, optimizer)

    def fit(self, samples):
        pass

    def predict(self, samples):
        return np.zeros((len(samples['states']),), dtype=np.float32)