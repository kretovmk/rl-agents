
import mpp as mp
import tensorflow as tf
import numpy as np
import logging

from multiprocessing import Process, Lock, Manager
from samplers.base import SamplerBase

logger = logging.getLogger('__main__')


class ParallelSampler(object):

    def __init__(self, n_workers, port, env_name, state_processor, max_steps):
        self.n_workers = n_workers
        self.env_name = env_name
        self.port = port
        self.state_processor = state_processor
        self.max_steps = max_steps
        self._launch_workers()

    def _launch_workers(self):
        processes = []
        for i in range(0, self.n_workers):
            cmd = 'python run_sampler.py {} {} {} {}'.format(self.env_name, self.n_workers, self.port, i)
            print('Executing ' + cmd)
            processes.append(runcmd(cmd))
        self.processes = processes

    def _collect_data(self):
        x = tf.Variable(0)
        with tf.Session() as sess:
            print 1
            #sess.run(tf.global_variables_initializer())
            print 2
            print sess.run(x)
            print 3

    def collect_samples(self, gamma, batch_size, sample=True):
        mgr = Manager()
        shared_list = mgr.list()
        shared_value = mgr.Value('i', 0)
        lock = Lock()
        jobs = []
        for i in xrange(self.n_workers):
            # p = Process(target=self._collect_data, args=(i, self.env, self.policy, self.state_processor, lock,
            #                                             shared_list, shared_value, batch_size, gamma, sample,
            #                                             self.max_steps))
            p = Process(target=self._collect_data, args=())
            p.start()
            jobs.append(p)
        for p in jobs: p.join()
        paths = shared_list
        states = np.concatenate([x['states'] for x in paths])
        actions = np.concatenate([x['actions'] for x in paths])
        prob_actions = np.concatenate([x['prob_actions'] for x in paths])
        returns = np.concatenate([x['returns'] for x in paths])
        return dict(states=states, actions=actions, returns=returns, prob_actions=prob_actions)
