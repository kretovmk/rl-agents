
import numpy as np
import tensorflow as tf


class NetworkBase(object):

    def __init__(self, scope, inp_shape, n_outputs):
        self.inp_shape = inp_shape
        self.n_outputs = n_outputs
        self.scope = scope
        self.inp, self.out, self.targets, self.loss = self._build_network()
        self.out_sq = tf.squeeze(self.out)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.loss)


    def _build_network(self):
        raise NotImplementedError

    def _iterate_minibatches(self, x, y, batch_size=32, shuffle=True):
        """Iterator over dataset."""
        l = x.shape[0]
        indices = np.arange(l)
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, l, batch_size):
            end_idx = min(start_idx + batch_size, l)
            if shuffle:
                excerpt = indices[start_idx:end_idx]
            else:
                excerpt = slice(start_idx, end_idx)
            yield x[excerpt], y[excerpt]

    def train(self, sess, x, y, n_epochs=1, batch_size=32, shuffle=True):
        losses = []
        for i in xrange(n_epochs):
            for batch_x, batch_y in self._iterate_minibatches(x, y, batch_size, shuffle):
                _, loss = sess.run([self.train_op, self.loss], feed_dict={self.inp: batch_x,
                                                                  self.targets: batch_y})
                losses.append(loss)
        return np.array(losses).mean()

    def predict_x(self, sess, x):
        return sess.run(self.out_sq, feed_dict={self.inp: [x]})

    def predict_batch(self, sess, batch_x):
        return sess.run(self.out, feed_dict={self.inp: batch_x})



