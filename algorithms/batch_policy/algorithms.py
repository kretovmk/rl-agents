
import time
import logging
import tensorflow as tf


logger = logging.getLogger('__main__')


class BatchPolicyBase(object):
    """
    Base object for batch optimization policy algorithms.
        Key ingredients:
            env -- environment
            state processor -- preprocessing of states
            policy -- policy to optimize
            baseline -- value to use in calculation of advantage function
            sampler -- collector of data from environment
            optimizer -- optimizer of policy (1st order, 2nd order etc.)
        Other attributes:
            gamma -- discounting factor
            state_shape -- shape of the state of environment
            n_actions -- number of available actions (discrete state space)
    """

    def __init__(self,
                 sess,
                 env,
                 state_processor,
                 gamma,
                 policy,
                 baseline,
                 sampler,
                 optimizer,
                 state_shape,
                 n_actions):
        self.sess = sess
        self.env = env
        self.state_processor = state_processor
        self.gamma = gamma
        self.policy = policy
        self.baseline = baseline
        self.sampler = sampler
        self.optimizer = optimizer
        self.state_shape = state_shape
        self.n_actions = n_actions
        self._init_variables()
        self.sess.run(tf.global_variables_initializer())
        logger.info('Agent variables initialized.')

    def _init_variables(self):
        raise NotImplementedError

    def collect_samples(self):
        return self.sampler.collect_samples()

    def process_samples(self, samples):
        return self.state_processor.process_samples(samples)

    def train(self, n_iter):
        logger.info('Start training.')
        for i in xrange(n_iter):
            itr_start_time = time.time()
            with logger.prefix('iter {} | '.format(i)):
                samples = self.collect_samples()









    def test(self):
        pass

