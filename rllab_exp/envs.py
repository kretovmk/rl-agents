
import cv2
import os
import gym
import numpy as np
import logging

from rllab.envs.base import Env, Step
from rllab.envs.gym_env import GymEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from sandbox.rocky.tf.envs.base import TfEnv
from gym import error
from gym import spaces
from gym.core import Env

NEW_SHAPE = (105, 80)

class TfEnvFrameProcessed(TfEnv):
    def observation_space(self):
        return self.wrapped_env.observation_space


class GymEnvFrameProcessed(GymEnv):
    def __init__(self, env_name):
        super(GymEnvFrameProcessed, self).__init__(env_name)
        self.observation_space = spaces.Box(low=0., high=1., shape=NEW_SHAPE)

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        next_obs = self._process_frame(next_obs)
        return Step(next_obs, reward, done, **info)

    def reset(self):
        if self._force_reset and self.monitoring:
            recorder = self.env._monitor.stats_recorder
            if recorder is not None:
                recorder.done = True
        obs = self.env.reset()
        return self._process_frame(obs)

    def _process_frame(self, frame):
        frame = cv2.resize(frame, *NEW_SHAPE)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = 255 - frame
        return frame / 255.
