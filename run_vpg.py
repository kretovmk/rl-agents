
import tensorflow as tf
import numpy as np
import logging
import gym
import os

from vpg.vpg_agent import VPGDense

"""
Vanilla Policy Gradient with function approximation and baseline.

Refs:
    1. http://rllab.readthedocs.io/en/latest/user/implement_algo_basic.html
    2. http://www.scholarpedia.org/article/Policy_gradient_methods

Limitations:
1. Discrete action space
2. Episodic tasks
"""

# general options
USE_CHECKPOINT = False    # loading from saved checkpoint if possible
ENV_NAME = 'CartPole-v0'   # gym's env name
MAX_ENV_STEPS = 1000   # limit for max steps during episode
CONCAT_LENGTH = 1  # should be >= 1 (mainly needed for concatenation Atari frames)
ENV_STATE_SHAPE = (4,)   # tuple
N_ACTIONS = 2   # int; only discrete action space
EXP_FOLDER = os.path.abspath("./experiments/{}".format(ENV_NAME))

# training options
NUM_ITER = 50000
BATCH_SIZE = 1000
EVAL_FREQ = 1   # evaluate every N env steps
RECORD_VIDEO_FREQ = 1000
GAMMA = 0.9
ADV = False


if __name__ == '__main__':
    tf.reset_default_graph()

    #logging
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    env = gym.make(ENV_NAME)

    # training
    with tf.Session() as sess:
        agent = VPGDense(hidden=(32,),
                         sess=sess,
                         env=env,
                         state_shape=ENV_STATE_SHAPE,
                         n_actions=N_ACTIONS,
                         state_processor=None,
                         adv=ADV,
                         gamma=GAMMA,
                         max_steps=MAX_ENV_STEPS)
        for i in xrange(NUM_ITER):
            states, actions, returns, av_reward = agent.run_batch_episodes(BATCH_SIZE)
            value_loss = 0.
            if ADV:
                for _ in xrange(10):
                    value_loss = agent.train_batch_value(sess, states, returns)
            policy_loss, step = agent.train_batch_policy(sess, states, actions, returns)
            if i % EVAL_FREQ == 0:
                res = agent.run_episode(sample=False)
                print 'Average undiscounted reward in iteration {} is {:.1f}, value loss is {:.1f}'.\
                    format(i, len(res['actions']), value_loss)
