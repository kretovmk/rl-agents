
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
from utils.tf_utils import cluster_spec

"""
General structure of cluster:
    - one parameter server -- make policy optimization step
    - N workers -- perform sampling of policy in environment
"""

flags = tf.flags

# general
flags.DEFINE_boolean('load_checkpoint', False, 'loading checkpoint')
flags.DEFINE_string('env', 'CartPole-v0', 'gym environment name')
flags.DEFINE_integer('concat_length', 1, 'concat len should be >= 1 (mainly needed for concatenation Atari frames)')
flags.DEFINE_integer('max_env_steps', 10000, 'max number of steps in environment')
flags.DEFINE_string('env_state_shape', '4', 'shape of env state')
flags.DEFINE_integer('n_actions', 2, 'number of actions')
flags.DEFINE_string('exp_folder', '.', 'folder with experiments')
flags.DEFINE_integer('n_workers', 4, 'number of workers')
# training
flags.DEFINE_integer('n_iter', 2, 'number of policy iterations')
flags.DEFINE_integer('batch_size', 1000, 'batch size policy sampling')
flags.DEFINE_integer('eval_freq', 1, 'frequency of evaluations')
flags.DEFINE_float('gamma', 0.9, 'discounting factor gamma')
# technical
flags.DEFINE_integer('port', 15100, 'starting port')
flags.DEFINE_integer('task', 0, 'number of task')

FLAGS = flags.FLAGS
env_state_shape = FLAGS.env_state_shape.split()


if __name__ == '__main__':
    tf.reset_default_graph()

    # launching parameter server -- also non-parallel calculations here
    spec = tf.train.ClusterSpec(cluster_spec(n_workers=FLAGS.n_workers, port=FLAGS.port))
    print 'Cluster spec: \n{}'.format(cluster_spec(n_workers=FLAGS.n_workers, port=FLAGS.port))
    server = tf.train.Server(spec, job_name='ps', task_index=0)
    print 'ps {} launched, server target: \"{}\"'.format(0, server.target)
    sess = tf.Session(server.target)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    env = gym.make(FLAGS.env_name)
    exp_folder = os.path.abspath((FLAGS.exp_folder + '/experiments/{}').format(FLAGS.env_name))
    checkpoint_dir, checkpoint_path, monitor_path = get_saver_paths(exp_folder)

    # training
    sess = tf.Session(server.target)

    worker_device = 'job:ps/task:0'
    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
        policy = NetworkCategorialDense(n_hidden=(16,),
                                        scope='policy',
                                        inp_shape=env_state_shape,
                                        n_outputs=FLAGS.n_actions)
        baseline_approximator = NetworkRegDense(n_hidden=(16,),
                                                scope='baseline',
                                                inp_shape=env_state_shape,
                                                n_outputs=1)
        baseline = NetworkBaseline(sess=sess,
                                   approximator=baseline_approximator,
                                   n_epochs=5,
                                   batch_size=32)
    #baseline = ZeroBaseline()
    state_processor = EmptyProcessor()
    # creating queue with state of ps server and workers: 1 -- sampling, 0 -- policy optimization
    with tf.device('job:ps/task:0'):
        queue_ps_state = tf.FIFOQueue(1, tf.int32, shapes=(), shared_name='queue_ps_state')
    ps_state_enqueue_op = queue_ps_state.enqueue(1)
    sess.run(ps_state_enqueue_op)
    # launching workers here
    parallel_worker = ParallelSampler(n_workers=FLAGS.n_workers,
                                      port=FLAGS.port,
                                      env_name=FLAGS.env_name,
                                      state_processor=state_processor,
                                      max_steps=FLAGS.max_env_steps)

    agent = TRPO(sess=sess,
                 gamma=FLAGS.gamma,
                 batch_size=FLAGS.batch_size,
                 policy=policy,
                 baseline=baseline,
                 sampler=parallel_worker,
                 monitor_path=monitor_path,
                 state_shape=env_state_shape,
                 n_actions=FLAGS.n_actions)

    saver = tf.train.Saver()
    if FLAGS.load_checkpoint:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)
            logger.info('Checkpoint {} loaded using specified path.'.format(latest_checkpoint))
        else:
            logger.info('No checkpoints were found. Starting from scratch.')

    agent.train_agent(n_iter=FLAGS.n_iter,
                      eval_freq=FLAGS.eval_freq,
                      saver=saver,
                      checkpoint_path=checkpoint_path)

    sess.close()
