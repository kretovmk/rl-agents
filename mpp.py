
import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp


N_WORKERS = 2


def cluster_spec(n_workers):
    cluster = {}
    port = 12222
    all_workers = []
    host = '127.0.0.1'
    for _ in range(n_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster

def sync(all_params, global_params):
    return tf.group(*[v1.assign(v2) for v1, v2 in zip(all_params, global_params)])


class Worker(object):

    def __init__(self, global_var, task, n_workers):
        self.global_var = global_var
        spec = cluster_spec(n_workers)
        cluster = tf.train.ClusterSpec(spec)
        self.config = tf.ConfigProto(device_filters=['/job:worker/task:{}/cpu:0'.format(task)])
        self.server = tf.train.Server(cluster, job_name='worker', task_index=task)
        self.var = tf.Variable(np.zeros(2), dtype=tf.float32, name='var_{}'.format(task))

        # TODO: add here sync instead of random


    def sampling(self, shared_list, arg):
        with tf.Session(self.server.target, config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(sync([self.var], [self.global_var]))
            res = self.var * arg
            res = sess.run(res)

        shared_list.append(res)




def parallel_sampler():

    inputs = [1, 1]

    global_var = tf.Variable(inputs, dtype=tf.float32, name='global_var')



    workers = []
    for i in xrange(N_WORKERS):
        workers.append(Worker(global_var, i, N_WORKERS))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

    jobs = []
    args = [1, 10]

    mgr = mp.Manager()
    shared_list = mgr.list()

    def sampling(shared_list):


        for task in xrange(N_WORKERS):

            p = mp.Process(target=workers[task].sampling, args=(shared_list, args[task],))
            p.start()
            jobs.append(p)

        for p in jobs: p.join()
        return shared_list

    for i in xrange(1000):
        t = time.time()
        sampling(shared_list)
        print shared_list
        del shared_list[:]
        print '******', time.time() - t


if __name__ == '__main__':

    parallel_sampler()
    #
    # inputs = np.array([1, 1])
    # for i in xrange(200):
    #     t = time.time()
    #     res = np.array(parallel_sampler(inputs))
    #     inputs = res.mean(axis=0)
    #     print res, time.time() - t