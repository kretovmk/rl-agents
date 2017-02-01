
import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp


def cluster_spec(n_workers):
    cluster = {}
    port = 12222
    host = '127.0.0.1'
    all_workers = []
    for _ in range(n_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    cluster['ps'] = ['{}:{}'.format(host, port)]
    return cluster


N_WORKERS = 4
SPEC = cluster_spec(N_WORKERS)


def sync(all_params, global_params):
    return tf.group(*[v1.assign(v2) for v1, v2 in zip(all_params, global_params)])


def run_ps_server():
    cluster = tf.train.ClusterSpec(SPEC)
    server = tf.train.Server(cluster, job_name="ps", task_index=0,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
    server.join()


def run_worker_server(task):
    cluster = tf.train.ClusterSpec(SPEC)
    server = tf.train.Server(cluster, job_name="worker", task_index=task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1,
                                                       inter_op_parallelism_threads=1))
    run_agent(task, server, cluster)


class AC(object):

    def __init__(self, task):
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(self.task)

        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                var = self._build()
                global_params = [var]

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                # Create policy and value networks
                self.var = self._build()

            all_params = [self.var]
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(all_params, global_params)])

    def _build(self):
        return tf.Variable(np.zeros(2))

    def start(self, sess):
        self.sess = sess
        self.sess.run(self.sync)


def run_agent(task, server, cluster):

    ac = AC(task)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(task)])
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_local_op = tf.variables_initializer(
        [v for v in tf.global_variables() if v.name.startswith("local")])
    init_all_op = tf.initialize_all_variables()

    def init_fn(ses):
        ses.run(init_all_op)

    sv = tf.train.Supervisor(is_chief=(task == 0),
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             ready_op=tf.report_uninitialized_variables(variables_to_save))

    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(init_local_op)
        ac.start(sess)
        train_config = None
        test_config = None

        if test_config is not None:
            assert train_config is not None

        episode = 0
        while not sv.should_stop() and episode <= 1:
            time.sleep(1)
            print 111
            episode += 1

    sv.stop()
    print 'worker {} stopped'.format(task)




if __name__ == '__main__':
    # start ps servers
    ps_worker = mp.Process(target=run_ps_server, args=())
    ps_worker.daemon = True
    ps_worker.start()

    # start workers
    workers = []
    for i in xrange(N_WORKERS):
        w = mp.Process(target=run_worker_server, args=(i,))
        w.daemon = True
        w.start()
        workers.append(w)

    for w in workers:
        w.join()

    ps_worker.terminate()