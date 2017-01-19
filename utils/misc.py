
import tensorflow as tf
import csv


def make_list(data):
    if not hasattr(data, '__iter__'):
        data = [data]
    return list(data)


def copy_parameters(sess, model_1, model_2):
    model_1_params = [t for t in tf.trainable_variables() if t.name.startswith(model_1.scope)]
    model_1_params = sorted(model_1_params, key=lambda v: v.name)
    model_2_params = [t for t in tf.trainable_variables() if t.name.startswith(model_2.scope)]
    model_2_params = sorted(model_2_params, key=lambda v: v.name)
    update_ops = []
    for p1, p2 in zip(model_1_params, model_2_params):
        op = p2.assign(p1)
        update_ops.append(op)
    sess.run(update_ops)


def write_csv(file_name, *arrays):
  with open(file_name, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for row in zip(*arrays):
      writer.writerow(row)