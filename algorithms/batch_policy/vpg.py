
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
        self.n_actions = n_actions
        self.clip_gradients = clip_gradients
        self.learning_rate = learning_rate
        super(VanillaPG, self).__init__(*args, **kwargs)

    def _init_variables(self):

        self.states_ph = self.policy.inp
        self.action_probs = self.policy.out
        self.actions_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='actions')
        self.advantages_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantages')

        self.actions_one_hot = tf.one_hot(self.actions_ph, depth=self.n_actions, dtype=tf.float32)
        self.likelihood = tf.reduce_sum(tf.multiply(self.action_probs, self.actions_one_hot), axis=1)
        self.policy_loss = -1. * tf.reduce_mean(tf.multiply(tf.log(self.likelihood), self.advantages_ph), axis=0)

        policy_grads = tf.gradients(self.policy_loss, self.policy.params)
        policy_grads, _ = tf.clip_by_global_norm(policy_grads, self.clip_gradients)
        self.policy_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_policy_op = self.policy_opt.apply_gradients(zip(policy_grads, self.policy.params), self.global_step)

        tf.summary.scalar("model/policy_loss", self.policy_loss)
        tf.summary.scalar("model/policy_grad_global_norm", tf.global_norm(policy_grads))
        tf.summary.scalar("model/policy_weights_global_norm", tf.global_norm(self.policy.params))


    def _optimize_policy(self, samples):
        adv = samples['returns'] - samples['baseline']
        states = samples['states']
        actions = samples['actions']
        feed_dict = {self.policy.inp: states,
                     self.actions_ph: actions,
                     self.advantages_ph: adv}
        summary, _, global_step, loss = self.sess.run([self.summary_op, self.train_policy_op, self.global_step,
                                              self.policy_loss], feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step)
        return loss, global_step
