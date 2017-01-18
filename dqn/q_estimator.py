
import tensorflow as tf
import os

# TODO: add testing period (evaluation)
# TODO: amend that ReplayMemory is for discrete and cont, and Q-learn is only for discrete


class QvalueEstimatorBase(object):
    """
    Base class for any estimators: value, Q-value, policy, advantage etc.
    """
    def __init__(self,
                 inp_shape=None,
                 n_actions=None,
                 scope='estimator',
                 sum_dir=None):
        self.inp_shape = inp_shape
        self.n_actions = n_actions
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(self.scope):
            self._build_model()
            if sum_dir:
                summary_folder = os.path.join(sum_dir, 'summary_{}'.format(scope))
                if not os.path.exists(sum_dir):
                    os.makedirs(sum_dir)
                self.summary_writer = tf.summary.FileWriter(summary_folder)

    def _build_model(self):
        self.state_ph = tf.placeholder(shape=(None,) + self.inp_shape, dtype=tf.float32, name='x')
        self.target_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='y')
        self.actions_ph = tf.placeholder(shape=(None, 1), dtype=tf.int32, name='actions')
        self.q_all = self._build_network()
        self.actions_one_hot = tf.one_hot(tf.squeeze(self.actions_ph, axis=1), self.n_actions, dtype=tf.float32)
        self.q = tf.reduce_sum(tf.mul(self.actions_one_hot, self.q_all), reduction_indices=[1])
        self.losses = tf.squared_difference(self.q, self.target_ph)
        self.loss = tf.reduce_mean(self.losses)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.histogram('loss_histogram', self.losses),
            tf.summary.histogram('q_values_histogram', self.q_all),
            tf.summary.scalar('max_q_value', tf.reduce_max(self.q_all))
        ])

    def predict_q_values(self, sess, states):
        return sess.run(self.q_all, feed_dict={self.state_ph: states})

    def update_step(self, sess, states, targets, actions):
        feed_dict = {self.state_ph: states, self.target_ph: targets, self.actions_ph: actions}
        summaries, global_step, loss, _ = sess.run([self.summaries, self.global_step,
                                                   self.loss, self.train_op], feed_dict=feed_dict)
        # summaries, loss, _ = sess.run([self.summaries,
        #                                             self.loss, self.train_op], feed_dict=feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

    def _build_network(self):
        raise NotImplementedError


class QvalueEstimatorConv(QvalueEstimatorBase):
    """
    Convolutional neural network for Atari games.
    """
    def __init__(self, *args, **kwargs):
        super(QvalueEstimatorConv, self).__init__(*args, **kwargs)

    def _build_network(self):
        conv1 = tf.contrib.layers.conv2d(inputs=self.state_ph, num_outputs=32, kernel_size=(8, 8), stride=(4, 4),
                                         padding='SAME', activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=(4, 4), stride=(2, 2),
                                         padding='SAME', activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=64, kernel_size=(3, 3), stride=(1, 1),
                                         padding='SAME', activation_fn=tf.nn.relu)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        out = tf.contrib.layers.fully_connected(fc1, self.n_actions)
        return out


class QvalueEstimatorDense(QvalueEstimatorBase):
    """
    Convolutional neural network for Atari games.
    """

    def __init__(self, *args, **kwargs):
        super(QvalueEstimatorDense, self).__init__(*args, **kwargs)

    def _build_network(self):
        flattened = tf.contrib.layers.flatten(self.state_ph)
        fc1 = tf.contrib.layers.fully_connected(flattened, 16)
        #fc2 = tf.contrib.layers.fully_connected(fc1, 32)
        out = tf.contrib.layers.fully_connected(fc1, self.n_actions)
        return out