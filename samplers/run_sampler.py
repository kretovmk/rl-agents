
import tensorflow as tf
import gym
import sys

from utils.tf_utils import cluster_spec

if __name__ == '__main__':
    # collecting input info
    env_name = sys.argv[1]
    n_workers, port, task = [int(x) for x in sys.argv[2:]]
    spec = tf.train.ClusterSpec(cluster_spec(n_workers, port))
    env = gym.make(env_name)
    state_shape = env.observaion_space.shape

    # launching worker
    server = tf.train.Server(spec, job_name='worker', task_index=task)
    sess = tf.Session(server.target)
    print 'worker {} launched, server target: \"{}\"'.format(task, server.target)

    # creating queues



