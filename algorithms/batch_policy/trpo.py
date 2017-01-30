
import logging
import tensorflow as tf
import numpy as np

from algorithms.batch_policy.base import BatchPolicyBase
from utils.tf_utils import flat_gradients, var_shape, SetFromFlat, GetFlat
from utils.math import conjugate_gradient

logger = logging.getLogger('__main__')


class TRPO(BatchPolicyBase):
    """
    TRPO with function approximation and baseline. No bootstrapping used.

    Refs:
        1. https://arxiv.org/abs/1502.05477
        2. https://github.com/wojzaremba/trpo/blob/master/main.py
        3. https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py
        4. RLLAB

    Limitations:
    1. Discrete action space
    2. Episodic tasks (but may work with non-episodic, like CartPole ).
    """
    def __init__(self, state_shape, n_actions, learning_rate=0.001, clip_gradients=10., *args, **kwargs):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.clip_gradients = clip_gradients
        self.learning_rate = learning_rate
        super(TRPO, self).__init__(*args, **kwargs)

    def _init_variables(self):

        tiny = 1e-6

        self.action_probs = self.policy.out
        self.prev_action_probs = tf.placeholder(shape=(None, self.n_actions), dtype=tf.int32, name='actions')
        self.states_ph = self.policy.inp
        self.actions_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='actions')
        self.advantages_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantages')

        self.actions_one_hot = tf.one_hot(self.actions_ph, depth=self.n_actions, dtype=tf.float32)
        self.cur_likelihood = tf.reduce_sum(tf.multiply(self.action_probs, self.actions_one_hot), axis=1)
        self.prev_likelihood = tf.reduce_sum(tf.multiply(self.prev_action_probs, self.actions_one_hot), axis=1)

        self.policy_loss = -1. * tf.reduce_mean(tf.divide(self.cur_likelihood, self.prev_likelihood) * \
                                                self.advantages_ph, axis=0)

        self.policy_vars = self.policy.params

        self.kl_div = tf.reduce_mean(self.prev_action_probs * \
                                tf.log(tf.divide(self.prev_action_probs + tiny, self.action_probs + tiny)))
        self.entropy = -1. * tf.reduce_mean(self.action_probs * tf.log(self.action_probs + tiny))
        self.losses = [self.policy_loss, self.kl_div, self.entropy]

        self.policy_grad = flat_gradients(self.policy_loss, self.policy_vars)
        self.kl_div_first_fixed = tf.reduce_mean(tf.stop_gradient(self.action_probs) * \
                            tf.log(tf.divide(tf.stop_gradient(self.action_probs) + tiny, self.action_probs + tiny)))

        self.kl_grads = flat_gradients(self.kl_div_first_fixed, self.policy_vars)
        shapes = map(var_shape, self.policy_vars)

        self.flat_tangent = tf.placeholder(shape=[None], dtype=tf.float32)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        self.grad_vec_prod = [tf.reduce_sum(g * t) for (g, t) in zip(self.kl_grads, tangents)]
        self.fisher_vec_prod = flat_gradients(self.grad_vec_prod, self.policy_vars)
        self.get_flat = GetFlat(self.sess, self.policy_vars)
        self.set_from_flat = SetFromFlat(self.sess, self.policy_vars)

        # self.policy_loss = -1. * tf.reduce_mean(tf.multiply(tf.log(self.likelihood), self.advantages_ph), axis=0)
        #
        # policy_grads = tf.gradients(self.policy_loss, self.policy.params)
        # policy_grads, _ = tf.clip_by_global_norm(policy_grads, self.clip_gradients)
        # self.policy_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # self.train_policy_op = self.policy_opt.apply_gradients(zip(policy_grads, self.policy.params), self.global_step)
        #
        # tf.summary.scalar("model/policy_loss", self.policy_loss)
        # tf.summary.scalar("model/policy_grad_global_norm", tf.global_norm(policy_grads))
        # tf.summary.scalar("model/policy_weights_global_norm", tf.global_norm(self.policy.params))


    def _optimize_policy(self, samples):

        cg_damping = 0.1
        max_kl = 0.01

        states = samples['states']
        actions = samples['actions']
        adv = samples['returns'] - samples['baseline']
        action_probs = samples['action_probs']
        feed_dict = {
            self.policy.inp: states,
            self.actions_ph: actions,
            self.advantages_ph: adv,
            self.prev_action_probs: action_probs
        }

        def fisher_vector_product(p):
            feed_dict[self.flat_tangent] = p
            return self.sess.run(self.fisher_vec_prod, feed_dict) + cg_damping * p

        grad = self.sess.run(self.policy_grad, feed_dict=feed_dict)
        step_direction = conjugate_gradient(fisher_vector_product, -grad)
        shs = .5 * step_direction.dot(fisher_vector_product(step_direction))
        step_max = np.sqrt(shs / max_kl)
        fullstep = step_direction / step_max
        neggdotstepdir = -grad.dot(step_direction)













        #adv = samples['returns'] - samples['baseline']
        #states = samples['states']
        #actions = samples['actions']
        #feed_dict = {self.policy.inp: states,
        #             self.actions_ph: actions,
        #             self.advantages_ph: adv}
        summary, _, global_step, loss = self.sess.run([self.summary_op, self.train_policy_op, self.global_step,
                                              self.policy_loss], feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step)
        return loss, global_step