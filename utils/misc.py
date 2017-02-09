
import tensorflow as tf
import numpy as np
import subprocess
import csv
import os

def runcmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    return p


def launch_workers(n_workers):
    processes = []
    for i in range(0, n_workers):
        cmd = 'python run_worker.py --task={}'.format(i)
        print('Executing ' + cmd)
        processes.append(runcmd(cmd))
    return processes


def make_list(data):
    if not hasattr(data, '__iter__'):
        data = [data]
    return list(data)


# only needed for q-learning
#def copy_parameters(sess, model_1, model_2):
#    model_1_params = [t for t in tf.trainable_variables() if t.name.startswith(model_1.scope)]
#    model_1_params = sorted(model_1_params, key=lambda v: v.name)
#    model_2_params = [t for t in tf.trainable_variables() if t.name.startswith(model_2.scope)]
#    model_2_params = sorted(model_2_params, key=lambda v: v.name)
#    update_ops = []
#    for p1, p2 in zip(model_1_params, model_2_params):
#        op = p2.assign(p1)
#        update_ops.append(op)
#    sess.run(update_ops)


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), 'Shape function assumes that shape is fully known.'
    return out


def flat_gradients(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(0, [tf.reshape(grad, [np.prod(var_shape(v))])
                       for (v, grad) in zip(var_list, grads)])


def write_csv(file_name, *arrays):
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in zip(*arrays):
            writer.writerow(row)



def get_saver_paths(exp_dir):
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(exp_dir, "monitor")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
    return checkpoint_dir, checkpoint_path, monitor_path
