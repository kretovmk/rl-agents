
import tensorflow as tf


floatX = tf.float32


def make_list(data):
    if not hasattr(data, '__iter__'):
        data = [data]
    return list(data)

