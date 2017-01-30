
import logging
import os
import gym
import tensorflow as tf

from utils.preprocessing import EmptyProcessor
from algorithms.batch_policy.trpo import TRPO
from baselines.universal import NetworkBaseline
from baselines.zero import ZeroBaseline
from networks.dense import NetworkCategorialDense, NetworkRegDense
from samplers.base import SamplerBase

# general options
USE_CHECKPOINT = False    # loading from saved checkpoint if possible
CONCAT_LENGTH = 1  # should be >= 1 (mainly needed for concatenation Atari frames)
ENV_NAME = 'CartPole-v0'   # gym's env name
MAX_ENV_STEPS = 1000   # limit for max steps during episode
ENV_STATE_SHAPE = (4,)   # tuple
N_ACTIONS = 2   # int; only discrete action space
EXP_FOLDER = os.path.abspath("temp/tf/experiments/{}".format(ENV_NAME))

# training options
NUM_ITER = 50000
BATCH_SIZE = 1000
EVAL_FREQ = 1   # evaluate every N env steps
RECORD_VIDEO_FREQ = 1000
GAMMA = 0.9
ADV = True


if __name__ == '__main__':
    tf.reset_default_graph()

    #logging
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    env = gym.make(ENV_NAME)

    # training
    with tf.Session() as sess:
        policy = NetworkCategorialDense(n_hidden=(16,),
                                        scope='policy',
                                        inp_shape=ENV_STATE_SHAPE,
                                        n_outputs=N_ACTIONS)
        state_processor = EmptyProcessor()
        baseline_approximator = NetworkRegDense(n_hidden=(16,),
                                                scope='baseline',
                                                inp_shape=ENV_STATE_SHAPE,
                                                n_outputs=1)
        baseline = NetworkBaseline(sess=sess,
                                   approximator=baseline_approximator,
                                   n_epochs=5,
                                   batch_size=32)
        #baseline = ZeroBaseline()
        sampler = SamplerBase(sess=sess,
                              env=env,
                              policy=policy,
                              state_processor=state_processor,
                              max_steps=MAX_ENV_STEPS)
        agent = TRPO(sess=sess,
                     gamma=GAMMA,
                     batch_size=BATCH_SIZE,
                     policy=policy,
                     baseline=baseline,
                     sampler=sampler,
                     log_dir=EXP_FOLDER,
                     state_shape=ENV_STATE_SHAPE,
                    n_actions=N_ACTIONS)
        agent.train_agent(n_iter=NUM_ITER,
                          eval_freq=EVAL_FREQ)

#/home/dd210/Desktop/rl-agents/tmp/tf/experiments/CartPole-v0

