
import tensorflow as tf
import numpy as np


def numel(x):
    return np.prod(var_shape(x))


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def flat_gradients(loss, var_list):
    """
    Same as tf.gradients but returns flat tensor.
    """
    grads = tf.gradients(loss, var_list)
    return tf.concat(values=[tf.reshape(grad, [np.prod(var_shape(v))]) for (v, grad) in zip(var_list, grads)], axis=0)


class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(shape=[total_size], dtype=tf.float32)
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(
                    v,
                    tf.reshape(
                        theta[
                            start:start +
                            size],
                        shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(values=[tf.reshape(v, [numel(v)]) for v in var_list], axis=0)

    def __call__(self):
        return self.op.eval(session=self.session)