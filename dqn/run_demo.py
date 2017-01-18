
import tensorflow as tf
import numpy as np
import logging
import gym
import os

from q_estimator import QvalueEstimatorDense
from dqn_agent import DQNAgent, RandomAgent
from utils import EmptyProcessor
from memory import ReplayMemory

"""
Q-learning with function approximation.

Following improvements to basic algorithm [https://webdocs.cs.ualberta.ca/~sutton/book/the-book-2nd.html]
are implemented:
1. DQN tricks [https://arxiv.org/abs/1312.5602]
2. Double DQN [https://arxiv.org/abs/1509.06461]
3. Duelling architecture [https://arxiv.org/abs/1511.06581]
4. Prioritized experience replay [https://arxiv.org/abs/1511.05952]
5. Optimality tightening [https://arxiv.org/abs/1611.01606]

Limitations:
1. Discrete action space
2. Episodic tasks
"""

# general options
USE_CHECKPOINT = False    # loading from saved checkpoint if possible
ENV_NAME = 'CartPole-v0'   # gym's env name
MAX_ENV_STEPS = 200   # limit for max steps during episode
CONCAT_STATES = 1  # should be >= 1 (mainly needed for concatenation Atari frames)
ENV_STATE_SHAPE = (4,)   # tuple
N_ACTIONS = 2   # int; only discrete action space
EXP_FOLDER = os.path.abspath("./experiments/{}".format(ENV_NAME))

# training options
NUM_EPISODES = 1000
EVAL_FREQ = 1000   # evaluate every N env steps
REPLAY_MEMORY_SIZE = 100000
REPLAY_MEMORY_SIZE_INIT = 10000
RECORD_VIDEO_FREQ = 1000
GAMMA = 0.9
UPD_TARGET_FREQ = 1000
EPS_START = 1.
EPS_END = 0.01
EPS_LAMBDA = 0.001
BATCH_SIZE = 32

# q-learning options
DOUBLE_DQN = True
DUELLING = True
OPT_TIGHT = True
PRIORITIZED = True

#logging
logging.basicConfig(filename=os.path.join(EXP_FOLDER, 'data.log'), level=logging.INFO)

assert CONCAT_STATES >= 1, 'CONCAT_STATES should be >= 1.'

if __name__ == '__main__':

    tf.reset_default_graph()

    env = gym.make(ENV_NAME)

    q_model = QvalueEstimatorDense(inp_shape=ENV_STATE_SHAPE,
                                   n_actions=N_ACTIONS,
                                   scope="q",
                                   sum_dir=EXP_FOLDER)

    target_model = QvalueEstimatorDense(inp_shape=ENV_STATE_SHAPE,
                                        n_actions=N_ACTIONS,
                                        scope="target_q",
                                        sum_dir=EXP_FOLDER)

    replay_memory = ReplayMemory(observation_shape=ENV_STATE_SHAPE,
                                 action_dim=1,
                                 max_steps=REPLAY_MEMORY_SIZE,
                                 observation_dtype=np.float32,
                                 action_dtype=np.int32,
                                 concat_observations=CONCAT_STATES > 1,
                                 concat_length=CONCAT_STATES)

    state_processor = EmptyProcessor()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        agent = DQNAgent(sess=sess,
                         env=env,
                         q_model=q_model,
                         target_model=target_model,
                         replay_memory=replay_memory,
                         state_processor=state_processor)

        exploration_agent = RandomAgent(N_ACTIONS)

        agent.fill_replay_memory(exploration_agent)

        for n_episode in xrange(NUM_EPISODES):
            if n_episode % EVAL_FREQ == 0:
                total_rewards = agent.run_episode(test=True)
                logging.INFO('Episode: {}, Reward: {}, Mode: Test'.format(n_episode, sum(total_rewards)))
            else:
                total_rewards = agent.run_episode(test=False)
                logging.INFO('Episode: {}, Reward: {}, Mode: Train'.format(n_episode, sum(total_rewards)))