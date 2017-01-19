import logging
import os

import tensorflow as tf

"""
TRPO with function approximation.

Based on code: https://github.com/tilarids/reinforcement_learning_playground/blob/master/trpo_agent.py

Link to article: https://arxiv.org/abs/1502.05477

Limitations:
1. Discrete action space
2. Episodic tasks
"""

# general options
USE_CHECKPOINT = False    # loading from saved checkpoint if possible
ENV_NAME = 'CartPole-v0'   # gym's env name
MAX_ENV_STEPS = 200   # limit for max steps during episode
CONCAT_LENGTH = 1  # should be >= 1 (mainly needed for concatenation Atari frames)
ENV_STATE_SHAPE = (4,)   # tuple
N_ACTIONS = 2   # int; only discrete action space
EXP_FOLDER = os.path.abspath("./experiments/{}".format(ENV_NAME))

# training options
NUM_EPISODES = 10000
EVAL_FREQ = 100   # evaluate every N env steps
RECORD_VIDEO_FREQ = 1000
GAMMA = 0.9
EPS_START = 1.
EPS_END = 0.01
EPS_LAMBDA = 0.001
BATCH_SIZE = 32

# TRPO options





if __name__ == '__main__':
    tf.reset_default_graph()

    #logging
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)