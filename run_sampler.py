
import tensorflow as tf
import numpy as np
import logging
import time
import gym
import sys
import os

from networks.dense import NetworkCategorialDense
from networks.conv import NetworkCategorialConvKeras
from utils.math import discount_rewards
from utils.tf_utils import cluster_spec
from utils.preprocessing import EmptyProcessor
from wrappers.envs import AtariStackFrames

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


####################---OPTIONS Atari---################################
# inp_shape = (210, 160, 3)
# proc_shape = (4, 84, 84)
# n_actions = 9
# STATE_PROCESSOR = EmptyProcessor(inp_state_shape=inp_shape,
#                                  proc_state_shape=proc_shape)
# POLICY = NetworkCategorialConvKeras(scope='policy',
#                                     inp_shape=STATE_PROCESSOR.proc_shape,
#                                     n_outputs=n_actions)
# ATARI_WRAPPER = True
#######################################################################

####################---OPTIONS Mountain car---#########################
# inp_shape = (2,)
# proc_shape = (2,)
# n_actions = 3
# STATE_PROCESSOR = EmptyProcessor(inp_state_shape=inp_shape,
#                                  proc_state_shape=proc_shape)
# POLICY = NetworkCategorialDense(n_hidden=(32,),
#                                 scope='policy',
#                                 inp_shape=STATE_PROCESSOR.proc_shape,
#                                 n_outputs=n_actions)
# ATARI_WRAPPER = False
#######################################################################

###################---OPTIONS Cartpole---#############################
inp_shape = (4,)
proc_shape = (4,)
n_actions = 2
STATE_PROCESSOR = EmptyProcessor(inp_state_shape=inp_shape,
                                 proc_state_shape=proc_shape)
POLICY = NetworkCategorialDense(n_hidden=(32,),
                                scope='policy',
                                inp_shape=STATE_PROCESSOR.proc_shape,
                                n_outputs=n_actions)
ATARI_WRAPPER = False
#######################################################################


def run_episode(sess, env, gamma, state_processor, max_steps, flatten_dim):
    states = []
    actions = []
    rewards = []
    prob_actions = []
    state = env.reset()
    state = state_processor.process(sess, state)
    for i in xrange(max_steps):
        probs = policy.predict_x(sess, state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        next_state, reward, terminal, _ = env.step(action)
        next_state = state_processor.process(sess, next_state)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        prob_actions.append(probs)
        state = next_state
        if terminal:
            break
    l = len(states)
    states = np.array(states).reshape((l, -1))
    actions = np.array(actions)
    rewards = np.array(rewards)
    prob_actions = np.array(prob_actions)
    returns = discount_rewards(np.array(rewards), gamma)
    res = np.concatenate((states, np.expand_dims(actions, 1), np.expand_dims(rewards, 1), prob_actions, np.expand_dims(returns, 1)), axis=1)
    assert flatten_dim == res.shape[1], 'Inconsistent flatten_dim and collected data by worker.'
    return res


if __name__ == '__main__':
    flatten_dim = np.prod(proc_shape) + 1 + 1 + n_actions + 1  # states, actions, rewards, prob_actions, returns
    # collecting input info
    env_name = sys.argv[1]
    gamma = float(sys.argv[2])
    n_workers, max_buf_size, max_steps, port, task = [int(x) for x in sys.argv[3:]]
    spec = tf.train.ClusterSpec(cluster_spec(n_workers, port))
    env = gym.make(env_name)
    if ATARI_WRAPPER:
        env = AtariStackFrames(env)
    state_processor = STATE_PROCESSOR

    # launching worker
    server = tf.train.Server(spec, job_name='worker', task_index=task)
    sess = tf.Session(server.target)
    print 'worker {} launched, server target: \"{}\"'.format(task, server.target)

    # creating queues
    with tf.device('job:ps/task:0'):
        queue_ps = tf.FIFOQueue(capacity=1, dtypes=tf.int32, shapes=(), shared_name='queue_ps')
        queue_sampled = tf.FIFOQueue(capacity=max_buf_size, dtypes=tf.float32,
                                     shapes=(flatten_dim,), shared_name='queue_sampled')
        queue_done = tf.FIFOQueue(capacity=max_buf_size, dtypes=tf.int32, shapes=(), shared_name='queue_done')
    logger.debug('Worker {}, queues ops created.'.format(task))

    # creating queue ops
    ph = tf.placeholder(tf.float32, shape=(None, flatten_dim))
    sampled_enq_op = queue_sampled.enqueue_many(ph)
    ps_deq_op = queue_ps.dequeue()
    done_enq_op = queue_done.enqueue(1)
    logger.debug('worker {}, queue ops created.'.format(task))

    # creating networks
    worker_device = "/job:worker/task:{}".format(task)
    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
        policy = POLICY
    logger.info('worker {} network built.'.format(task))

    while len(sess.run(tf.report_uninitialized_variables())) > 0:
        logger.info('worker {}:variables not initialized, sleeping.'.format(task))
        time.sleep(1)

    while True:
        sess.run(ps_deq_op)
        res = run_episode(sess, env, gamma, state_processor, max_steps, flatten_dim)
        sess.run(sampled_enq_op, feed_dict={ph: res})
        sess.run(done_enq_op)
