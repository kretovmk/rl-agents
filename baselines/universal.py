
from baselines.base import BaselineBase


class NetworkBaseline(BaselineBase):

    def __init__(self, sess, approximator, n_epochs, batch_size):
        self.sess = sess
        self.approximator = approximator
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        super(NetworkBaseline, self).__init__()

    def fit(self, samples):
        x = samples['states']
        y = samples['returns']
        loss = self.approximator.train(self.sess, x, y, self.n_epochs, self.batch_size)
        return loss

    def predict_value(self, samples):
        x = samples['states']
        samples['baseline'] = self.approximator.predict_batch(self.sess, x)
        return samples