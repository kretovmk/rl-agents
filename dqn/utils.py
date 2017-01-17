
import tensorflow as tf

from collections import namedtuple


floatX = tf.float32
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def make_list(data):
    if not hasattr(data, '__iter__'):
        data = [data]
    return list(data)

