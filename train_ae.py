
import tensorflow as tf
import numpy as np
import gym

from wrappers.envs import AtariStackFrames



if __name__ == '__main__':

    env = AtariStackFrames(gym.make('MsPacman-v0'))
    

