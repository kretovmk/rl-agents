

class BaselineBase(object):

    def __init__(self):
        pass

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError