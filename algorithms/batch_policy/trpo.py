
import logging
import tensorflow as tf
import numpy as np

from algorithms.batch_policy.base import BatchPolicyBase
from utils.tf_utils import flat_gradients, var_shape, SetFromFlat, GetFlat
from utils.math import conjugate_gradient, line_search, line_search_expected_improvement

logger = logging.getLogger('__main__')


# TODO: check parameters of line search -- now 0.5.. replace with 0.9?
# TODO: make profiling: if conj grad can be replaced with inverse transoformation of matrix
# TODO: replace 2nd derivatives with another def of FIM
# TODO: check line search algo with / without expected improvement rate
# TODO: make loading keras model and adding name scope to name of variables if possible
# TODO: saving checpoints + saving policy as keras model


class TRPO(BatchPolicyBase):
    """
    TRPO with function approximation and baseline. No bootstrapping used.

    Refs:
        1. https://arxiv.org/abs/1502.05477
        2. https://github.com/wojzaremba/trpo/blob/master/main.py
        3. https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py
        4. https://github.com/openai/rllab

    Limitations:
    1. Discrete action space
    2. Episodic tasks (but may work with non-episodic, like CartPole).
    """
    def __init__(self, state_shape, n_actions, learning_rate=0.001, entropy_coeff=0.001, subsampling=0.1,
                 clip_gradients=10., *args, **kwargs):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.entropy_coeff = entropy_coeff
        self.subsampling = subsampling
        self.clip_gradients = clip_gradients
        self.learning_rate = learning_rate
        super(TRPO, self).__init__(*args, **kwargs)

    def _init_variables(self):

        tiny = 1e-6  # for numerical stability

        self.action_probs = self.policy.out
        self.entropy_coeff_ph = tf.placeholder(shape=(), dtype=tf.float32, name='entropy_coeff')
        self.prev_action_probs = tf.placeholder(shape=(None, self.n_actions), dtype=tf.float32, name='actions')
        self.states_ph = self.policy.inp
        self.actions_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='actions')
        self.advantages_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantages')

        self.actions_one_hot = tf.one_hot(self.actions_ph, depth=self.n_actions, dtype=tf.float32)
        self.cur_likelihood = tf.reduce_sum(tf.multiply(self.action_probs, self.actions_one_hot), axis=1)
        self.prev_likelihood = tf.reduce_sum(tf.multiply(self.prev_action_probs, self.actions_one_hot), axis=1)

        self.entropy = -1. * tf.reduce_mean(self.action_probs * tf.log(self.action_probs + tiny))

        self.policy_loss = -1. * tf.reduce_mean(tf.divide(self.cur_likelihood, self.prev_likelihood) *
                                                self.advantages_ph, axis=0) - self.entropy * self.entropy_coeff_ph

        self.policy_vars = self.policy.params

        self.kl_div = tf.reduce_mean(self.prev_action_probs * \
                                tf.log(tf.divide(self.prev_action_probs + tiny, self.action_probs + tiny)))
        self.losses = [self.policy_loss, self.kl_div, self.entropy]

        self.policy_grad = flat_gradients(self.policy_loss, self.policy_vars)
        self.kl_div_first_fixed = tf.reduce_mean(tf.stop_gradient(self.action_probs) * \
                            tf.log(tf.divide(tf.stop_gradient(self.action_probs) + tiny, self.action_probs + tiny)))

        self.kl_grads = tf.gradients(self.kl_div_first_fixed, self.policy_vars)
        shapes = map(var_shape, self.policy_vars)

        self.flat_tangent = tf.placeholder(shape=[None], dtype=tf.float32, name='flat_tangent')
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

        tf.summary.scalar("model/policy_loss", self.policy_loss)
        tf.summary.scalar("model/policy_grad_global_norm", tf.global_norm([self.policy_grad]))
        tf.summary.scalar("model/policy_weights_global_norm", tf.global_norm(self.policy.params))
        tf.summary.scalar("model/policy_entropy", tf.global_norm([self.entropy]))


    def _optimize_policy(self, samples):
        cg_damping = 0.1
        max_kl = 0.01

        states = samples['states']
        actions = samples['actions']
        adv = samples['returns'] - samples['baseline']
        action_probs = samples['prob_actions']

        # TODO: check normalization
        adv -= adv.mean()
        adv /= (adv.std() + 1e-6)

        # full dataset -- for calculation of policy gradient
        feed_dict = {
            self.entropy_coeff_ph: self.entropy_coeff,
            self.policy.inp: states,
            self.actions_ph: actions,
            self.advantages_ph: adv,
            self.prev_action_probs: action_probs
        }

        # subsampled dataset -- for approximate calculation of gradient step direction
        n_samples = int(len(states) * self.subsampling)
        ix_samples = np.random.choice(range(n_samples), size=n_samples, replace=False)
        feed_dict_ss = {
            self.entropy_coeff_ph: self.entropy_coeff,
            self.policy.inp: states[ix_samples],
            self.actions_ph: actions[ix_samples],
            self.advantages_ph: adv[ix_samples],
            self.prev_action_probs: action_probs[ix_samples]
        }

        # calc on the basis of subsampled
        def fisher_vector_product(p):
            feed_dict_ss[self.flat_tangent] = p
            return self.sess.run(self.fisher_vec_prod, feed_dict_ss) + cg_damping * p

        prev_params = self.get_flat()
        # accurate calc of gradient
        grad = self.sess.run(self.policy_grad, feed_dict=feed_dict)
        # approximate calc of step direction
        step_direction = conjugate_gradient(fisher_vector_product, -grad)
        shs = 0.5 * step_direction.dot(fisher_vector_product(step_direction))
        step_max = np.sqrt(shs / max_kl)
        fullstep = step_direction / step_max
        neggdotstepdir = -grad.dot(step_direction)

        def loss(th):
            self.set_from_flat(th)
            return self.sess.run(self.losses[0], feed_dict=feed_dict)

        new_params = line_search_expected_improvement(loss, prev_params, fullstep, neggdotstepdir / step_max)
        self.set_from_flat(new_params)

        # approximate calculation of loss, kl_div, entropy
        loss, kl_div, entropy = self.sess.run(self.losses, feed_dict=feed_dict_ss)
        if kl_div > 2.0 * max_kl:
            self.set_from_flat(prev_params)
            logger.info('New parameters caused too big KL divergence: {:.4f}; backed up to old parameters'\
                        .format(kl_div))

        summary, global_step = self.sess.run([self.summary_op, self.global_step], feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step)
        return loss, global_step