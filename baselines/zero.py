
import numpy as np

from baselines.base import BaselineBase


class ZeroBaseline(BaselineBase):

    def __init__(self):
        super(ZeroBaseline, self).__init__()

    def fit(self):
        pass

    def predict(self, samples):
        return np.zeros((len(samples['states']),), dtype=np.float32)