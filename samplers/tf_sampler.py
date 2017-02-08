
import tensorflow as tf
import numpy as np
import logging
import gym
import os

from utils.math import discount_rewards
from wrappers.envs import AtariStackFrames
from utils.misc import runcmd

logger = logging.getLogger('__main__')


class ParallelSampler(object):

    def __init__(self, sess, policy, max_buf_size, batch_size, n_workers, port, env_name, n_actions, state_processor, max_steps, gamma):
        self.sess = sess
        self.policy = policy
        self.max_buf_size = max_buf_size
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.env_name = env_name
        self.n_actions = n_actions
        self.port = port
        self.state_processor = state_processor
        self.max_steps = max_steps
        self.gamma = gamma
        self.env =  gym.make(env_name)
        self.env = AtariStackFrames(self.env)

        # creating queues
        flatten_dim = np.prod(state_processor.proc_shape) + 1 + 1 + n_actions + 1  # states, act, rew, prob_act, ret
        with tf.device('job:ps/task:0'):
            self.queue_ps = tf.FIFOQueue(capacity=1, dtypes=tf.int32, shapes=(), shared_name='queue_ps')
            self.queue_sampled = tf.FIFOQueue(capacity=max_buf_size, dtypes=tf.float32,
                                         shapes=(flatten_dim,), shared_name='queue_sampled')
            self.queue_done = tf.FIFOQueue(capacity=max_buf_size, dtypes=tf.int32, shapes=(), shared_name='queue_done')
        # queue operations
        self._build_queue_ops()
        # launch worker servers
        self.processes = self._launch_workers()

    def _build_queue_ops(self):
        # get size
        self.sampled_size_op = self.queue_sampled.size()
        self.ps_size_op = self.queue_ps.size()
        self.done_size_op = self.queue_done.size()
        # put in queue
        self.ps_enq_op = self.queue_ps.enqueue(1)
        # clear queue
        self.ph = tf.placeholder(tf.int32, shape=())
        self.sampled_deq_op = self.queue_sampled.dequeue_many(self.ph)
        self.done_deq_op = self.queue_done.dequeue_many(self.ph)

    def _launch_workers(self):
        processes = []
        for i in range(0, self.n_workers):
            cmd = 'python run_sampler.py {} {} {} {} {} {} {}'.format(self.env_name, self.gamma, self.n_workers,
                                                                self.max_buf_size, self.max_steps, self.port, i)
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
                #print self.sess.run(self.done_size_op), tasks_given
                if self.sess.run(self.done_size_op) == tasks_given:
                    # getting samples
                    n_samples = self.sess.run(self.sampled_size_op)
                    raw_samples = self.sess.run(self.sampled_deq_op, feed_dict={self.ph: n_samples})
                    states, actions, rewards, prob_actions, returns = self._unflatten_array(raw_samples)
                    # clearing queues
                    self.sess.run(self.done_deq_op, feed_dict={self.ph: tasks_given})
                    break
        logger.info('Performed {} tasks by workers.'.format(tasks_given))
        logger.info('Collected {} samples'.format(n_samples))
        return dict(states=states, actions=actions, returns=returns, prob_actions=prob_actions)

    def _unflatten_array(self, ar):
        # states, actions, rewards, prob_actions, returns
        l = len(ar)
        d = np.array([np.prod(self.state_processor.proc_shape), 1,  1, self.n_actions, 1]).cumsum()
        states = ar[:, :d[0]].reshape((l,) + self.state_processor.proc_shape)
        actions = ar[:, d[0]]
        rewards = ar[:, d[1]: d[2]]
        prob_actions = ar[:, d[2]: d[3]]
        returns = ar[:, d[3]]
        return states, actions, rewards, prob_actions, returns

    def run_episode(self, gamma, sample=True):
        states = []
        actions = []
        rewards = []
        prob_actions = []
        state = self.env.reset()
        state = self.state_processor.process(self.sess, state)
        for i in xrange(self.max_steps):
            probs = self.policy.predict_x(self.sess, state)
            if sample:
                action = np.random.choice(np.arange(len(probs)), p=probs)
            else:
                action = np.argmax(probs)
            next_state, reward, terminal, _ = self.env.step(action)
            next_state = self.state_processor.process(self.sess, next_state)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            prob_actions.append(probs)
            state = next_state
            if terminal:
                break
        returns = discount_rewards(np.array(rewards), gamma)
        return dict(states=states,
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    prob_actions=np.array(prob_actions),
                    returns=returns)

    def test_agent(self, sample=False, sess=None):
        res = self.run_episode(gamma=1., sample=sample)
        total_reward = res['returns'][0]
        episode_length = len(res['returns'])
        tf.summary.scalar("test/total_reward", total_reward)
        tf.summary.scalar("test/episode_length", episode_length)
        return total_reward, episode_length
