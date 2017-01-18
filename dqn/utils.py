
import tensorflow as tf


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


class AtariImgProcessor(object):
    """
    Processes raw Atari image. Resize and convert to grayscale.
    """
    def __init__(self,
                 img_shape=(210, 160,3),
                 bounding_box=(34, 0, 160, 160),
                 resize_shape=(84, 84)):
        with tf.variable_scope('state_processor'):
            self.inp = tf.placeholder(shape=img_shape, dtype=tf.float32)
            self.out = tf.image.rgb_to_grayscale(self.inp)
            self.out = tf.image.crop_to_bounding_box(self.out, *bounding_box)
            self.out = tf.image.resize_images(self.out, *resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            #tf.to_float(self.X_pl) / 255.0

    def process(self, sess, image):
        return sess.run(self.out, feed_dict={self.inp: image})


class EmptyProcessor(object):
    """
    Does nothing.
    """
    def __init__(self):
        pass

    def process(self, sess, state):
        return state

