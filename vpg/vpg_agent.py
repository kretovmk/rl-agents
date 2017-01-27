
import tensorflow as tf
import itertools
import numpy as np

from utils.math import discount_rewards

# TODO: check how cutting off trajectories spoil final result (it causes incorrect returns in the end)
# TODO: check how baseling should be fitted -- now hardcoded 100 epochs
# TODO: add tf summary


class VPGBase(object):

    def __init__(self, sess,
                       env,
                       state_shape,
                       n_actions,
                       state_processor,
                       adv=True,
                       lr=0.001,
                       clip_gradients=10.,
                       gamma=0.99,
                       max_steps=200):
        self.sess = sess
        self.env = env
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.state_processor = state_processor
        self.gamma = gamma
        self.max_steps = max_steps
        self.adv = adv
        self.lr = lr
        self.clip_gradients = clip_gradients

        self.states_ph = tf.placeholder(shape=(None,) + self.state_shape, dtype=tf.float32)
        self.actions_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.returns_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.value_targets_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.action_probs = self._build_policy_network()
        self.value = self._build_value_network()
        self.policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy')
        self.value_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value')

        # value
        self.value_loss = tf.reduce_mean(tf.squared_difference(self.value, self.value_targets_ph), axis=0)
        value_grads = tf.gradients(self.value_loss, self.value_params)
        value_grads, _ = tf.clip_by_global_norm(value_grads, self.clip_gradients)
        self.value_opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_value_op = self.value_opt.apply_gradients(zip(value_grads, self.value_params))

        # policy
        self.actions_one_hot = tf.one_hot(self.actions_ph, depth=self.n_actions, dtype=tf.float32)
        self.likelihood = tf.reduce_sum(tf.multiply(self.action_probs, self.actions_one_hot), axis=1)
        if self.adv:
            self.policy_loss = -1. * tf.reduce_mean(tf.log(self.likelihood) * (self.returns_ph - self.value), axis=0)
        else:
            self.policy_loss = -1. * tf.reduce_mean(tf.log(self.likelihood) * self.returns_ph, axis=0)
        policy_grads = tf.gradients(self.policy_loss, self.policy_params)
        policy_grads, _ = tf.clip_by_global_norm(policy_grads, self.clip_gradients)
        self.policy_opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_policy_op = self.policy_opt.apply_gradients(zip(policy_grads, self.policy_params), self.global_step)

        self.sess.run(tf.global_variables_initializer())

    def train_batch_policy(self, sess, states, actions, returns):
        feed_dict = {self.states_ph: states,
                     self.actions_ph: actions,
                     self.returns_ph: returns}
        _, global_step, loss = sess.run([self.train_policy_op, self.global_step, self.policy_loss],
                                        feed_dict=feed_dict)
        return loss, global_step

    def train_batch_value(self, sess, states, targets):
        feed_dict = {self.states_ph: states,
                     self.value_targets_ph: targets}
        _, loss = sess.run([self.train_value_op, self.value_loss],
                                                  feed_dict=feed_dict)
        return loss

    def _build_policy_network(self):
        """
        Create variables in scope 'policy'.
        """
        raise NotImplementedError

    def _build_value_network(self):
        """
        Create variables in scope 'value'.
        """
        raise NotImplementedError

    def run_episode(self, sample=True):
        states = []
        actions = []
        rewards = []
        state = self.env.reset()
        for i in xrange(self.max_steps):

            action = self.choose_action(state, sample)
            #print action, i
            next_state, reward, terminal, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if terminal:
                break
        returns = discount_rewards(np.array(rewards), self.gamma)
        return dict(states=np.array(states),
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    returns=returns)

    def choose_action(self, state, sample=True):
        prob_actions = self.sess.run(self.action_probs, feed_dict={self.states_ph: [state]})[0]
        if sample:
            return np.random.choice(np.arange(len(prob_actions)), p=prob_actions)
        else:
            return np.argmax(prob_actions)

    def run_batch_episodes(self, batch_size, sample=True):
        paths = []
        prev_len = 0
        for t in itertools.count():
            paths.append(self.run_episode(sample))
            cur_len = sum([len(x['actions']) for x in paths])
            if cur_len >= batch_size:
                if t == 0:
                    print 'WARNING: just one truncated trajectory was used for batch. Consider increasing batch size.'

                ix = cur_len - prev_len
                for k, v in paths[-1].iteritems():
                    paths[-1][k] = v[:-ix]
                break
            prev_len = cur_len
        av_reward = np.array([x['rewards'].sum() for x in paths]).mean()
        states = np.concatenate([x['states'] for x in paths])
        actions = np.concatenate([x['actions'] for x in paths])
        returns = np.concatenate([x['returns'] for x in paths])
        return states, actions, returns, av_reward


class VPGDense(VPGBase):

    def __init__(self, hidden, *args, **kwargs):
        self.hidden = hidden
        super(VPGDense, self).__init__(*args, **kwargs)

    def _build_policy_network(self):
        with tf.variable_scope('policy'):
            out = tf.contrib.layers.flatten(self.states_ph)
            for n_hid in self.hidden:
                out = tf.contrib.layers.fully_connected(out, n_hid)
                out = tf.nn.relu(out)
            out = tf.contrib.layers.fully_connected(out, self.n_actions)
            out = tf.nn.softmax(out)
        return out

    def _build_value_network(self):
        with tf.variable_scope('value'):
            out = tf.contrib.layers.flatten(self.states_ph)
            for n_hid in self.hidden:
                out = tf.contrib.layers.fully_connected(out, n_hid)
                out = tf.nn.tanh(out)
            out = tf.contrib.layers.fully_connected(out, 1)
        return tf.squeeze(out)
