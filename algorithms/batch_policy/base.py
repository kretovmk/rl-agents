
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
                 gamma,
                 batch_size,
                 policy,
                 baseline,
                 sampler,
                 optimizer):
        self.sess = sess
        self.gamma = gamma
        self.batch_size = batch_size
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
        return self.sampler.process_samples(samples)

    def train_agent(self, n_iter, eval_freq):
        logger.info('Started training.')
        for i in xrange(n_iter):
            logger.info('\n\n' + '*'*40 + '\n' + 'Iteration {} \n'.format(i))

            start_time = time.time()
            samples = self.sampler.collect_samples(self.gamma, self.batch_size)
            logger.info('Collected samples, took {:.2f} sec'.format(time.time() - start_time))

            start_time = time.time()
            self.baseline.fit(samples)
            logger.info('Fitted baseline, took {:.2f} sec'.format(time.time() - start_time))

            start_time = time.time()
            self._optimize_policy()
            logger.info('Optimized policy, took {:.2f} sec'.format(time.time() - start_time))

            if n_iter % eval_freq == 0:
                self.test_agent(sample=False)

    def test_agent(self, sample):
        total_reward, episode_length = self.sampler.run_episode(sample)
        return total_reward, episode_length

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



