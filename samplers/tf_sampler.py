
import itertools
import time
import tensorflow as tf
import numpy as np
import logging

from samplers.base import SamplerBase
from utils.misc import runcmd

logger = logging.getLogger('__main__')


class ParallelSampler(object):

    def __init__(self, sess, max_buf_size, batch_size, n_workers, port, env_name, state_processor, max_steps):
        self.sess = sess
        self.max_buf_size = max_buf_size
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.env_name = env_name
        self.port = port
        self.state_processor = state_processor
        self.max_steps = max_steps
        # creating queues
        with tf.device('job:ps/task:0'):
            self.queue_ps = tf.FIFOQueue(capacity=1, dtypes=tf.int32, shapes=(), shared_name='queue_ps')
            self.queue_sampled = tf.FIFOQueue(capacity=max_buf_size, dtypes=tf.float32,
                                         shapes=state_processor.proc_shape, shared_name='queue_sampled')
            self.queue_done = tf.FIFOQueue(capacity=max_buf_size, dtypes=tf.int32, shapes=(), shared_name='queue_done')

        # queue operations
        self._build_queue_ops()
        # launch worker servers
        self.processes = self._launch_workers()

    def _build_queue_ops(self):
        # get size
        self.sampled_size_op = self.queue_ps.size()
        self.ps_size_op = self.queue_ps.size()
        self.done_size_op = self.queue_done.size()
        # put in queue
        self.ps_enq_op = self.queue_ps.enqueue(1)
        # clear queue
        self.ph = tf.placeholder(tf.int32, shape=())
        self.sampled_deq_op = self.queue_sampled.dequeue_many(self.ph)
        self.



    def _launch_workers(self):
        processes = []
        for i in range(0, self.n_workers):
            cmd = 'python run_sampler.py {} {} {} {}'.format(self.env_name, self.n_workers, self.port, i)
            print('Executing command: ' + cmd)
            processes.append(runcmd(cmd))
        return processes

    def collect_samples(self, gamma, batch_size, sample=True):
        tasks_given = 0
        while True:
            n_samples = self.sess.run(self.sampled_size_op)
            ps_state = self.sess.run(self.ps_size_op)
            if n_samples < self.batch_size and ps_state == 0:
                self.sess.run(self.ps_enq_op)
                tasks_given += 1
            elif n_samples >= self.batch_size:
                if self.sess.run(self.done_size_op) == tasks_given:
                    # getting samples
                    n_samples = self.sess.run(self.sampled_size_op)
                    raw_samples = self.sess.run(self.sampled_deq_op, feed_dict={self.ph: n_samples})
                    states, actions, prob_actions, returns = self._convert_array(raw_samples)
                    # clearing queues




    def _convert_array(self, ar):
        pass
        # TODO: .........
