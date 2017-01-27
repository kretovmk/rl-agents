
import numpy as np


class RandomAgent(object):
    """
    Random uniform sampling from discrete action space.
    """
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def choose_action(self, state):
        return np.random.randint(0, self.n_actions)