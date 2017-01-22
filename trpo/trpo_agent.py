
import tensorflow as tf
import numpy as np
import logging

from utils.math import cat_sample, line_search, discount_rewards
from utils.misc import var_shape, flat_gradients

logger = logging.getLogger('__main__')

# TODO: 1. compare conj. gradients with inversing matrix
# TODO: compare self-made CG with scipy realization
# TODO: check line search. looks not optimal...?
# TODO: check discount reward time -- self.made realization and through scipy signal

DTYPE = tf.float32


class TRPOAgent(object):
    """
    Class for agent governed by TRPO.
    """
    def __init__(self, sess,
                       env,
                       state_shape,
                       policy,
                       value,
                       state_processor,
                       gamma=0.99
                 ):
        self.sess = sess
        self.env = env
        self.state_shape = state_shape
        self.policy = policy
        self.value = value
        self.state_processor = state_processor
        self.gamma = gamma

        self.state_ph = tf.placeholder(shape=(None,) + self.state_shape, dtype=DTYPE, name='states')

        # TODO: prev_obs, prev_action -- why needed?

        self.cur_action_1h = tf.placeholder(shape=(None, self.env.action_space.n), dtype=DTYPE, name='cur_actions')
        #self.prev_action_1h = tf.placeholder(shape=(None, self.env.action_space.n), dtype=DTYPE, name='prev_actions')
        self.advantages = tf.placeholder(shape=(None,), dtype=DTYPE, name='advantages')
        self.prev_policy = tf.placeholder(shape=(None, self.env.action_space.n), dtype=DTYPE, name='prev_policy')

        # TODO: why 'returns'.. ? Seems like were not used

        self.loss = -1. * tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(
                    self.cur_action_1h * tf.div(self.policy, self.prev_policy)), 1) * self.advantages)

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        # Calculating policy step !!!

        var_list = tf.trainable_variables()

        # TODO: check how this works without eval
        def get_variables_flat_form():
          op = tf.concat(
              0, [tf.reshape(v, [np.prod(var_shape(v))]) for v in var_list])
          return op.eval(session=self.session)
        self.get_variables_flat_form = get_variables_flat_form


