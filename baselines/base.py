



class BaselineBase(object):

    def __init__(self, approximator, optimizer):
        self.approximator = approximator
        self.optimizer = optimizer

    def fit(self, samples):
        raise NotImplementedError

    def predict(self, samples):
        raise NotImplementedError