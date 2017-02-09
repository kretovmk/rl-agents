
import os
import gym
import keras
import signal
import logging
import tensorflow as tf

from utils.misc import get_saver_paths
from utils.preprocessing import EmptyProcessor
from algorithms.batch_policy.trpo import TRPO
from baselines.universal import NetworkBaseline
from baselines.zero import ZeroBaseline
from networks.dense import NetworkCategorialDense, NetworkRegDense
from networks.conv2 import NetworkCategorialConvKerasPretrained, NetworkRegConvKerasPretrained
from networks.conv import NetworkCategorialConvKeras, NetworkRegConvKeras
from samplers.tf_sampler import ParallelSampler
from utils.tf_utils import cluster_spec
from wrappers.envs import AtariStackFrames

"""
General structure of cluster:
    - one parameter server -- make policy optimization step
    - N workers -- perform sampling of policy in environment
"""

#####################################################################
##########################--OPTIONS START--##########################

flags = tf.flags

# general
flags.DEFINE_boolean('load_checkpoint', False, 'loading checkpoint')
flags.DEFINE_string('env_name', 'MsPacman-v0', 'gym environment name')
flags.DEFINE_boolean('atari_wrapper', True, 'gym environment name')
flags.DEFINE_integer('max_env_steps', 30, 'max number of steps in environment')
flags.DEFINE_integer('n_actions', 9, 'number of actions')
flags.DEFINE_string('exp_folder', '.', 'folder with experiments')
flags.DEFINE_integer('n_workers', 1, 'number of workers')
flags.DEFINE_float('subsampling', 0.1, 'subsampling for appr calc of 2nd derivatives')
# training
flags.DEFINE_integer('n_iter', 1000, 'number of policy iterations')
flags.DEFINE_integer('batch_size', 50, 'batch size policy sampling')
flags.DEFINE_integer('eval_freq', 1, 'frequency of evaluations')
flags.DEFINE_float('gamma', 0.99, 'discounting factor gamma')
flags.DEFINE_integer('baseline_epochs', 3, 'epochs when fitting baseline')
flags.DEFINE_float('baseline_batch_size', 64, 'batch size for fitting baseline')
# technical
flags.DEFINE_integer('port', 15100, 'starting port')
flags.DEFINE_integer('task', 0, 'number of task')

FLAGS = flags.FLAGS

env_proc_state_shape = (12, 105, 80)
env_inp_state_shape = (12, 105, 80)
logging_level = logging.INFO
STATE_PROCESSOR = EmptyProcessor(inp_state_shape=env_inp_state_shape,
                                 proc_state_shape=env_proc_state_shape)

fn = 'model_epoch99.h5'

###########################--OPTIONS END--###########################
#####################################################################

def load_keras_model(fn):
    return keras.models.load_model(fn)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    tf.reset_default_graph()

    # launching parameter server -- also non-parallel calculations here
    spec = tf.train.ClusterSpec(cluster_spec(n_workers=FLAGS.n_workers, port=FLAGS.port))
    logger.info('Cluster spec: \n{}'.format(cluster_spec(n_workers=FLAGS.n_workers, port=FLAGS.port)))
    server = tf.train.Server(spec, job_name='localhost', task_index=0)
    logger.info('Parameter server {} launched, server target: \"{}\"'.format(0, server.target))
    sess = tf.Session(server.target)

    # environment
    env = gym.make(FLAGS.env_name)
    env = AtariStackFrames(env)
    env_inp_state_shape = env.observation_space.shape
    logger.debug('Env observation original state shape: {}'.format(env_inp_state_shape))
    logger.debug('Env observation processed state shape: {}'.format(env_proc_state_shape))

    # checkpoint paths
    exp_folder = os.path.abspath((FLAGS.exp_folder + '/experiments/{}').format(FLAGS.env_name))
    checkpoint_dir, checkpoint_path, monitor_path = get_saver_paths(exp_folder)
    logger.debug('checkpoint_dir: {}\n checkpoint_path: {}\n monitor_path: {}'\
                 .format(checkpoint_dir, checkpoint_path, monitor_path))




    # initializing session and components
    sess = tf.Session(server.target)
    worker_device = 'job:localhost/task:0'

    # temp
    # pol = load_keras_model(fn)
    # val = load_keras_model(fn)
    # with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
    #     inp_pol = tf.placeholder(tf.float32, shape=(None,)+STATE_PROCESSOR.proc_shape)
    #     out_pol = pol(inp_pol)
    #     inp_val = tf.placeholder(tf.float32, shape=(None,)+STATE_PROCESSOR.proc_shape)
    #     out_val = pol(inp_val)

    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device, ps_device=worker_device)):
        #print 'keras sess set'
        policy = NetworkCategorialConvKerasPretrained('policy',
                                                      inp_shape=STATE_PROCESSOR.proc_shape,
                                                      n_outputs=FLAGS.n_actions,
                                                      fn=fn)
        baseline_approximator = NetworkRegConvKerasPretrained('baseline',
                                                              inp_shape=STATE_PROCESSOR.proc_shape,
                                                              n_outputs=1,
                                                              fn=fn)
        baseline = NetworkBaseline(sess=sess,
                                   approximator=baseline_approximator,
                                   n_epochs=FLAGS.baseline_epochs,
                                   batch_size=FLAGS.baseline_batch_size)
        #baseline = ZeroBaseline()

    # launching parallel workers for sampling
    max_buf_size = FLAGS.batch_size + FLAGS.max_env_steps * FLAGS.n_workers * 2
    parallel_sampler = ParallelSampler(sess=sess,
                                       policy=policy,
                                       max_buf_size=max_buf_size,
                                       batch_size=FLAGS.batch_size,
                                       n_workers=FLAGS.n_workers,
                                       port=FLAGS.port,
                                       env_name=FLAGS.env_name,
                                       n_actions=FLAGS.n_actions,
                                       state_processor=STATE_PROCESSOR,
                                       max_steps=FLAGS.max_env_steps,
                                       gamma=FLAGS.gamma,
                                       atari_wrapper=FLAGS.atari_wrapper)

    # creating agent
    agent = TRPO(sess=sess,
                 gamma=FLAGS.gamma,
                 batch_size=FLAGS.batch_size,
                 policy=policy,
                 baseline=baseline,
                 sampler=parallel_sampler,
                 monitor_path=monitor_path,
                 state_shape=STATE_PROCESSOR.proc_shape,
                 n_actions=FLAGS.n_actions,
                 subsampling=FLAGS.subsampling)

    # loading variables from checkpoint if applicable
    saver = tf.train.Saver()
    if FLAGS.load_checkpoint:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)
            logger.info('Checkpoint {} loaded using specified path.'.format(latest_checkpoint))
        else:
            logger.info('No checkpoints were found. Starting from scratch.')

    # training agent
    agent.train_agent(n_iter=FLAGS.n_iter,
                      eval_freq=FLAGS.eval_freq,
                      saver=saver,
                      checkpoint_path=checkpoint_path)

    # closing session and killing the workers
    for p in parallel_sampler.processes:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    sess.close()
