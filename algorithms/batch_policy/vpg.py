
import itertools
import logging
import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model
from algorithms.batch_policy.base import BatchPolicyBase
from utils.math import discount_rewards

# TODO: check how cutting off trajectories spoil final result (it causes incorrect returns in the end)
# TODO: check how baseling should be fitted -- now hardcoded 100 epochs
# TODO: add tf summary

logger = logging.getLogger('__main__')


class VanillaPG(BatchPolicyBase):
    """
    Vanilla Policy Gradient with function approximation and baseline. No bootstrapping used.

    Refs:
        1. http://rllab.readthedocs.io/en/latest/user/implement_algo_basic.html
        2. http://www.scholarpedia.org/article/Policy_gradient_methods

    Limitations:
    1. Discrete action space
    2. Episodic tasks (but may work with non-episodic, like CartPole ).
    """
    def __init__(self, state_shape, n_actions, learning_rate=0.001, clip_gradients=10., *args, **kwargs):
        self.state_shape = state_shape
        self.n_action = n_actions
        self.clip_gradients = clip_gradients
        self.learning_rate = learning_rate
        super(VanillaPG, self).__init__(*args, **kwargs)

    def _init_variables(self):

        self.states_ph = self.policy.inp
        self.action_probs = self.policy.out
        self.actions_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='actions')
        self.advantages_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantages')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.actions_one_hot = tf.one_hot(self.actions_ph, depth=self.policy.n_a, dtype=tf.float32)
        self.likelihood = tf.reduce_sum(tf.multiply(self.action_probs, self.actions_one_hot), axis=1)
        self.policy_loss = -1. * tf.reduce_mean(tf.multiply(tf.log(self.likelihood), self.advantages_ph), axis=0)

        policy_grads = tf.gradients(self.policy_loss, self.policy.params)
        policy_grads, _ = tf.clip_by_global_norm(policy_grads, self.clip_gradients)
        self.policy_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_policy_op = self.policy_opt.apply_gradients(zip(policy_grads, self.policy.params), self.global_step)

        tf.summary.scalar("model/policy_loss", self.policy_loss)
        tf.summary.scalar("model/policy_grad_global_norm", tf.global_norm(policy_grads))
        tf.summary.scalar("model/policy_var_global_norm", tf.global_norm(self.policy.params))


    def _optimize_policy(self, samples):
        adv = samples['returns'] - samples['baseline']
        states = samples['states']
        actions = samples['actions']
        feed_dict = {self.policy.inp: states,
                     self.actions_ph: actions,
                     self.advantages_ph: adv}
        _, global_step, loss = self.sess.run([self.train_policy_op, self.global_step, self.policy_loss],
                                             feed_dict=feed_dict)
        return loss, global_step





#-----------------------------TEMP to be deleted

        # self.states_ph = tf.placeholder(shape=(None,) + self.state_shape, dtype=tf.float32, name='states')
        # self.actions_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='actions')
        # self.advantages_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantages')
        # self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # self.action_probs = self.policy.build_network(self.states_ph)
        #self.value = self._build_value_network()
        # self.policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy')
        # self.value_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value')

        # value
        # self.value_loss = tf.reduce_mean(tf.squared_difference(self.value, self.value_targets_ph), axis=0)
        # value_grads = tf.gradients(self.value_loss, self.value_params)
        # value_grads, _ = tf.clip_by_global_norm(value_grads, self.clip_gradients)
        # self.value_opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        # self.train_value_op = self.value_opt.apply_gradients(zip(value_grads, self.value_params))

        # policy
        #self.actions_one_hot = tf.one_hot(self.actions_ph, depth=self.n_actions, dtype=tf.float32)
        #self.likelihood = tf.reduce_sum(tf.multiply(self.action_probs, self.actions_one_hot), axis=1)
        # if self.adv:
        #     self.policy_loss = -1. * tf.reduce_mean(tf.log(self.likelihood) * (self.returns_ph - self.value), axis=0)
        # else:
        #     self.policy_loss = -1. * tf.reduce_mean(tf.log(self.likelihood) * self.returns_ph, axis=0)
        # policy_grads = tf.gradients(self.policy_loss, self.policy_params)
        # policy_grads, _ = tf.clip_by_global_norm(policy_grads, self.clip_gradients)
        # self.policy_opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        # self.train_policy_op = self.policy_opt.apply_gradients(zip(policy_grads, self.policy_params), self.global_step)
        # tf.summary.scalar("model/policy_loss", pi_loss / bs)
        # tf.summary.scalar("model/value_loss", vf_loss / bs)
        # tf.summary.scalar("model/entropy", entropy / bs)
        # tf.summary.image("model/state", pi.x)
        # tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
        # tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))


#     def train_batch_policy(self, sess, states, actions, returns):
#         feed_dict = {self.states_ph: states,
#                      self.actions_ph: actions,
#                      self.returns_ph: returns}
#         _, global_step, loss = sess.run([self.train_policy_op, self.global_step, self.policy_loss],
#                                         feed_dict=feed_dict)
#         return loss, global_step
#
#     def train_batch_value(self, sess, states, targets):
#         feed_dict = {self.states_ph: states,
#                      self.value_targets_ph: targets}
#         _, loss = sess.run([self.train_value_op, self.value_loss],
#                                                   feed_dict=feed_dict)
#         return loss
#
#     def _build_policy_network(self):
#         """
#         Create variables in scope 'policy'.
#         """
#         raise NotImplementedError
#
#     def _build_value_network(self):
#         """
#         Create variables in scope 'value'.
#         """
#         raise NotImplementedError
#
#     def run_episode(self, sample=True):
#         states = []
#         actions = []
#         rewards = []
#         state = self.env.reset()
#         for i in xrange(self.max_steps):
#
#             action = self.choose_action(state, sample)
#             #print action, i
#             next_state, reward, terminal, _ = self.env.step(action)
#             states.append(state)
#             actions.append(action)
#             rewards.append(reward)
#             state = next_state
#             if terminal:
#                 break
#         returns = discount_rewards(np.array(rewards), self.gamma)
#         return dict(states=np.array(states),
#                     actions=np.array(actions),
#                     rewards=np.array(rewards),
#                     returns=returns)
#
#     def choose_action(self, state, sample=True):
#         prob_actions = self.sess.run(self.action_probs, feed_dict={self.states_ph: [state]})[0]
#         if sample:
#             return np.random.choice(np.arange(len(prob_actions)), p=prob_actions)
#         else:
#             return np.argmax(prob_actions)
#
#     def run_batch_episodes(self, batch_size, sample=True):
#         paths = []
#         prev_len = 0
#         for t in itertools.count():
#             paths.append(self.run_episode(sample))
#             cur_len = sum([len(x['actions']) for x in paths])
#             if cur_len >= batch_size:
#                 if t == 0:
#                     print 'WARNING: just one truncated trajectory was used for batch. Consider increasing batch size.'
#
#                 ix = cur_len - prev_len
#                 for k, v in paths[-1].iteritems():
#                     paths[-1][k] = v[:-ix]
#                 break
#             prev_len = cur_len
#         av_reward = np.array([x['rewards'].sum() for x in paths]).mean()
#         states = np.concatenate([x['states'] for x in paths])
#         actions = np.concatenate([x['actions'] for x in paths])
#         returns = np.concatenate([x['returns'] for x in paths])
#         return states, actions, returns, av_reward
#
#
#
# class VPGDense(VPGBase):
#
#     def __init__(self, hidden, *args, **kwargs):
#         self.hidden = hidden
#         super(VPGDense, self).__init__(*args, **kwargs)
#
#     def _build_policy_network(self):
#         inputs = Input(shape=self.state_shape, name='policy_input')
#         out = inputs
#         for i, n_hid in enumerate(self.hidden):
#             out = Dense(output_dim=n_hid, activation='relu', name='dense_policy_{}'.format(i))(out)
#         out = Dense(output_dim=self.n_actions, activation='softmax', name='policy_softmax')(out)
#         model = Model(input=inputs, output=out)
#         policy = model(self.states_ph)
#         return policy
#
#     def _build_value_network(self):
#         inputs = Input(shape=self.state_shape, name='value_input')
#         out = inputs
#         for i, n_hid in enumerate(self.hidden):
#             out = Dense(output_dim=n_hid, activation='relu', name='dense_value_{}'.format(i))(out)
#         out = Dense(output_dim=1, name='value_out')(out)
#         model = Model(input=inputs, output=out)
#         value = model(self.states_ph)
#         return tf.squeeze(value)


#av_reward = np.array([x['rewards'].sum() for x in paths]).mean()