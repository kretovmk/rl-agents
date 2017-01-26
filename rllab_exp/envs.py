
import os
import cv2
import gym
import numpy as np

#from gym.monitoring import monitor_manager
from rllab.envs.base import Env, Step
from rllab.envs.gym_env import GymEnv, FixedIntervalVideoSchedule, NoVideoSchedule, \
    CappedCubicVideoSchedule, convert_gym_space
from rllab.core.serializable import Serializable
from rllab.misc import logger
from gym.spaces import Box


NEW_SHAPE = (105, 80)


class GymEnvMod(GymEnv):
    def __init__(self, env_name, record_video=True, video_schedule=None, log_dir=None, record_log=True,
                 force_reset=False):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        env = gym.envs.make(env_name)
        env.observation_space = Box(low=0., high=1., shape=NEW_SHAPE)
        self.env = env
        self.env_id = env.spec.id

        #monitor_manager.logger.setLevel(logging.WARNING)

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        #self._observation_space = convert_gym_space(env.observation_space)
        self._observation_space = convert_gym_space(env.observation_space)
        self._action_space = convert_gym_space(env.action_space)
        self._horizon = env.spec.timestep_limit
        self._log_dir = log_dir
        self._force_reset = force_reset


    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        next_obs = self._process_frame(next_obs)
        return Step(next_obs, reward, done, **info)

    def reset(self):
        if self._force_reset and self.monitoring:
            recorder = self.env._monitor.stats_recorder
            if recorder is not None:
                recorder.done = True
        obs = self._process_frame(self.env.reset())
        return obs

    def _process_frame(self, frame):
        frame = cv2.resize(frame, NEW_SHAPE)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = 255 - frame
        return np.expand_dims(frame, 0) / 255.