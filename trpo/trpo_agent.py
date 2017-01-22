
import tensorflow as tf
import numpy as np
import logging
import itertools

from utils.math import cat_sample, line_search, discount_rewards, conjugate_gradient
from utils.misc import var_shape, flat_gradients

logger = logging.getLogger('__main__')

# TODO: 1. compare conj. gradients with inversing matrix
# TODO: compare self-made CG with scipy realization
# TODO: check line search. looks not optimal...?
# TODO: check discount reward time -- self.made realization and through scipy signal

DTYPE = tf.float32
TOL = 1e-8


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
                       gamma=0.99,
                       max_steps=200
                 ):
        self.sess = sess
        self.env = env
        self.state_shape = state_shape
        self.policy = policy
        self.value = value
        self.state_processor = state_processor
        self.gamma = gamma
        self.max_steps = max_steps

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
          return op.eval(session=self.sess)
        self.get_variables_flat_form = get_variables_flat_form

        # define a function to set all trainable variables from a flat tensor theta.
        def create_set_variables_from_flat_form_function():
            shapes = map(var_shape, var_list)
            total_size = sum(np.prod(shape) for shape in shapes)
            theta_in = tf.placeholder(DTYPE, [total_size])
            start = 0
            assigns = []
            for (shape, v) in zip(shapes, var_list):
                size = np.prod(shape)
                assigns.append(tf.assign(v, tf.reshape(theta_in[start:start + size], shape)))
                start += size
            op = tf.group(*assigns)

            def set_variables_from_flat_form(theta):
                return self.sess.run(op, feed_dict={theta_in: theta})

            return set_variables_from_flat_form

        self.set_variables_from_flat_form = create_set_variables_from_flat_form_function()

        self.policy_gradients_op = flat_gradients(self.loss, var_list)

        # TODO: check why do we need stop gradient for prev_policy here
        # TODO: check if this expression can only been used for on-policy, otherwise re-do.........
        # TODO: this is an average of woj and tilar code...
        self.kl_div = tf.reduce_sum(tf.stop_gradient(self.policy) *
                                    tf.log(tf.div(tf.stop_gradient(self.policy) + TOL,
                                                  self.policy + TOL))) / tf.cast(tf.shape(self.state_ph)[0], DTYPE)

        kl_div_grad_op = tf.gradients(self.kl_div, var_list)

        self.flat_mult = tf.placeholder(DTYPE, shape=[None])


        # Do the actual multiplication. Some shape shifting magic.
        start = 0
        multiplier_parts = []
        for var in var_list:
          shape = var_shape(var)
          size = np.prod(shape)
          part = tf.reshape(self.flat_mult[start:(start + size)], shape)
          multiplier_parts.append(part)
          start += size

        product_op_list = [tf.reduce_sum(kl_derivation * multiplier) for
                           (kl_derivation, multiplier) in zip(kl_div_grad_op, multiplier_parts)]

        # Second derivation
        self.fisher_product_op_list = flat_gradients(product_op_list, var_list)


    def run_episode(self):

        action_1h = np.zeros(self.env.action_space.n)
        states, actions, rewards, action_probs, actions_one_hot = [], [], [], [], []

        state = self.env.reset()

        for t in itertools.count():


            action_probs = self.sess.run(self.policy, feed_dict={self.state_ph: np.expand_dims(state, 0)})
            action = int(cat_sample(action_probs)[0])
            action_1h = 0
            action_1h[action] = 1

            states.append(state)
            actions.append(action)
            action_probs.append(action_probs)
            actions_one_hot.append(action_1h)

            next_state, reward, terminal, _ = self.env.step(action)

            rewards.append(reward)
            state = next_state

            if t == self.max_steps or terminal:
                path = {'states': states,
                        'actions': actions,
                        'rewards': rewards,
                        'action_probs': action_probs,
                        'actions_one_hot': actions_one_hot}
                return path


    def train(self, batch_size=8, n_iter=100):

        for i in xrange(n_iter):

            paths = []
            for _ in xrange(batch_size):
                path = self.run_episode()
                paths.append(path)

            for path in paths:
                path['baseline'] = self.value.predict(path['states'])
                path['returns'] = discount_rewards(path['rewards'], self.gamma)
                path['advantages'] = [x - y for x, y in zip(path['returns'], path['baseline'])]

            # TODO: check validation of value losses, advantages normalization (tilar)


            advant = np.concatenate([path['advantages'] for path in paths])
            advant -= advant.mean()
            advant /= (advant.std() + TOL)

            actions = np.concatenate([path['actions_one_hot'] for path in paths])


            # TODO: add option for with / without bootstrapping here
            prev_policy = np.concatenate([path['action_probs'] for path in paths])
            value_loss = self.value.fit(paths)
            print i, value_loss


            previous_parameters_flat = self.get_variables_flat_form()

            feed_dict = {self.state_ph: np.concatenate([path['states'] for path in paths]),
                         self.advantages: advant,
                         self.cur_action_1h : actions,
                         self.prev_policy: prev_policy}


            def fisher_vector_product(multiplier):
                feed_dict[self.flat_mult] = multiplier
                conjugate_gradients_damping = 0.1
                return self.sess.run(self.fisher_product_op_list, feed_dict) + conjugate_gradients_damping * multiplier


            policy_gradients = self.sess.run(self.policy_gradients_op, feed_dict)

            step_direction = conjugate_gradient(fisher_vector_product, -policy_gradients)


            hessian_vector_product = step_direction.dot(fisher_vector_product(step_direction))
            max_kl = 0.01


            # This is our \beta.
            max_step_length = np.sqrt(2 * max_kl / hessian_vector_product)
            max_step = max_step_length * step_direction


            def get_loss_for(weights_flat):
                self.set_variables_from_flat_form(weights_flat)
                loss = self.sess.run(self.loss, feed_dict)
                kl_divergence = self.sess.run(self.kl_div, feed_dict)
                if kl_divergence > max_kl:
                    logger.info("Hit the safeguard: %s", kl_divergence)
                    return float('inf')
                else:
                    return loss


            # search along the search direction.
            new_weights = line_search(get_loss_for, previous_parameters_flat, max_step)

            self.set_variables_from_flat_form(new_weights)

            mean_path_len = np.mean([len(path['rewards']) for path in paths])
            print 'mean path len: {}'.format(mean_path_len)









































