
import logging
import os
import gym
import tensorflow as tf

from utils.misc import get_saver_paths
from utils.preprocessing import EmptyProcessor
from algorithms.batch_policy.trpo import TRPO
from baselines.universal import NetworkBaseline
from baselines.zero import ZeroBaseline
from networks.dense import NetworkCategorialDense, NetworkRegDense
from samplers.mp_sampler import ParallelSampler
from workers.base import WorkerBase

# general options
LOAD_CHECKPOINT = False    # loading from saved checkpoint if possible
CONCAT_LENGTH = 1  # should be >= 1 (mainly needed for concatenation Atari frames)
ENV_NAME = 'CartPole-v0'   # gym's env name
MAX_ENV_STEPS = 1000   # limit for max steps during episode
ENV_STATE_SHAPE = (4,)   # tuple
N_ACTIONS = 2   # int; only discrete action space
EXP_FOLDER = '.'   # 'temp/tf/experiments'
N_WORKERS = 4    # number of cpu workers

# training options
NUM_ITER = 50000
BATCH_SIZE = 100000
EVAL_FREQ = 1   # evaluate every N env steps
RECORD_VIDEO_FREQ = 1000
GAMMA = 0.9



if __name__ == '__main__':
    tf.reset_default_graph()

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    env = gym.make(ENV_NAME)
    exp_folder = os.path.abspath((EXP_FOLDER + '/experiments/{}').format(ENV_NAME))
    checkpoint_dir, checkpoint_path, monitor_path = get_saver_paths(exp_folder)

    # training
    sess = tf.Session()
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
    parallel_worker = ParallelSampler(n_workers=N_WORKERS,
                                      env=env,
                                      policy=policy,
                                      state_processor=state_processor,
                                      max_steps=MAX_ENV_STEPS)

    agent = TRPO(sess=sess,
                 gamma=GAMMA,
                 batch_size=BATCH_SIZE,
                 policy=policy,
                 baseline=baseline,
                 sampler=parallel_worker,
                 monitor_path=monitor_path,
                 state_shape=ENV_STATE_SHAPE,
                 n_actions=N_ACTIONS)

    saver = tf.train.Saver()
    if LOAD_CHECKPOINT:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)
            logger.info('Checkpoint {} loaded using specified path.'.format(latest_checkpoint))
        else:
            logger.info('No checkpoints were found. Starting from scratch.')

    agent.train_agent(n_iter=NUM_ITER,
                      eval_freq=EVAL_FREQ,
                      saver=saver,
                      checkpoint_path=checkpoint_path)

    sess.close()

#/home/dd210/Desktop/rl-agents/tmp/tf/experiments/CartPole-v0

