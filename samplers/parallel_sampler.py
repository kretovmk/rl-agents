
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import itertools
import logging

from utils.math import discount_rewards
from samplers.base import SamplerBase
from copy import copy

logger = logging.getLogger('__main__')


class ParallelSampler(object):

    def __init__(self, n_workers, samplers):
        self.n_workers = n_workers
        self.samplers = samplers

    def worker_run_episode(self, sampler, lock, shared_list, shared_value, batch_size, gamma, sample):
        while shared_value.value < batch_size:
            print shared_value
            res = sampler.run_episode(gamma, sample)
            shared_list.append(res)
            with lock:
                shared_value.value += len(res['states'])

    def collect_samples(self, gamma, batch_size, sample=True):
        mgr = mp.Manager()
        shared_list = mgr.list()
        shared_value = mgr.Value('i', 0)
        lock = mp.Lock()
        jobs = []
        for i in xrange(self.n_workers):
            p = mp.Process(target=self.worker_run_episode, args=(self.samplers[i], lock, shared_list, shared_value,
                                                                 batch_size, gamma, sample))
            p.start()
            jobs.append(p)
        for p in jobs: p.join()
        paths = shared_list
        states = np.concatenate([x['states'] for x in paths])
        actions = np.concatenate([x['actions'] for x in paths])
        prob_actions = np.concatenate([x['prob_actions'] for x in paths])
        returns = np.concatenate([x['returns'] for x in paths])
        return dict(states=states, actions=actions, returns=returns, prob_actions=prob_actions)
