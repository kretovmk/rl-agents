
from baselines.base import BaselineBase


class NetworkBaseline(BaselineBase):

    def __init__(self, approximator=None):
        self.approximator = approximator
        super(NetworkBaseline, self).__init__()

    def fit(self, sess, samples, n_epochs=1, batch_size=32):
        x = samples['states']
        y = samples['returns']
        loss = self.approximator.train(sess, x, y, n_epochs, batch_size)
        return loss

    def predict_value(self, sess, samples):
        x = samples['states']
        self.approximator.predict(sess, x)