
import tensorflow as tf


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

    def process_frame(self, sess, image):
        return sess.run(self.out, feed_dict={self.inp: image})