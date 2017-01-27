
import time
import logging
import itertools
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
    """

    def __init__(self,
                 sess,
                 env,
                 state_processor,
                 gamma,
                 policy,
                 baseline,
                 sampler,
                 optimizer):
        self.sess = sess
        self.env = env
        self.state_processor = state_processor
        self.gamma = gamma
        self.policy = policy
        self.baseline = baseline
        self.sampler = sampler
        self.optimizer = optimizer
        self._init_variables()
        self.sess.run(tf.global_variables_initializer())
        logger.info('Agent variables initialized.')

    def collect_samples(self):
        return self.sampler.collect_samples()

    def process_samples(self, samples):
        return self.state_processor.process_samples(samples)

    def train_agent(self, n_iter, eval_freq):
        logger.info('Started training.')
        for i in xrange(n_iter):

            logger.info('\n\nIteration {} \n'.format(i))

            start_time = time.time()
            raw_samples = self.collect_samples()
            logger.info('Collected raw samples, took {:.2f} sec'.format(time.time() - start_time))

            start_time = time.time()
            samples = self.process_samples(raw_samples)
            logger.info('Processed samples, took {:.2f} sec'.format(time.time() - start_time))

            start_time = time.time()
            self.baseline.fit(samples)
            logger.info('Fitted baseline, took {:.2f} sec'.format(time.time() - start_time))

            start_time = time.time()
            self._optimize_policy()
            logger.info('Optimized policy, took {:.2f} sec'.format(time.time() - start_time))

            if n_iter % eval_freq == 0:
                self.test_agent()

    def test_agent(self, sample=False):
        state = self.env.reset()
        total_reward = 0.
        for t in itertools.count():
            action = self.policy.choose_action(state, sample)
            state, reward, terminal, _ = self.env.step(action)
            total_reward += reward
            if terminal:
                break
        return total_reward, t

    def _init_variables(self):
        """
        Initialize variables and build computational graph.
        """
        raise NotImplementedError

    def _optimize_policy(self):
        """
        Optimize policy on the basis of data collected.
        """
        raise NotImplementedError



