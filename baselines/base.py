

class BaselineBase(object):

    def __init__(self):
        pass

    def fit(self, samples):
        raise NotImplementedError

    def predict_value(self, samples):
        raise NotImplementedError