
import numpy as np

from baselines.base import BaselineBase


class ZeroBaseline(BaselineBase):

    def __init__(self):
        super(ZeroBaseline, self).__init__()

    def fit(self, samples):
        pass

    def predict_value(self, samples):
        samples['baseline'] = np.zeros((len(samples['states']),), dtype=np.float32)
        return samples