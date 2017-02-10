

import os
#os.system('pip install keras')
import cv2
import gym
import keras
import logging
import numpy as np



#from gym.monitoring import monitor_manager
from rllab.envs.base import Env, Step
from rllab.envs.gym_env import GymEnv, FixedIntervalVideoSchedule, NoVideoSchedule, \
    CappedCubicVideoSchedule, convert_gym_space
from rllab.core.serializable import Serializable
from rllab.misc import logger
from gym.spaces import Box
from collections import deque


NEW_SHAPE = (105, 80)


class GymEnvMod(GymEnv):
    def __init__(self, env_name, record_video=False, video_schedule=None, log_dir=None, record_log=True,
                 force_reset=False, n_frames=4):
        full_model = keras.models.load_model('model_epoch29.h5')
        self.n_frames = n_frames
        self.premodel = keras.models.Model(input=full_model.layers[0].input, output=full_model.layers[-3].output)
        #keras.backend.set_learning_phase(0)
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        env = gym.envs.make(env_name)
        #env.observation_space = Box(low=0., high=1., shape=NEW_SHAPE)
        env.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(256,))
        self.env = env
        self.env_id = env.spec.id

        self._checkpoint_buffer = []
        self.buffer = deque(maxlen=self.n_frames)

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

        self._observation_space = convert_gym_space(env.observation_space)
        self._action_space = convert_gym_space(env.action_space)
        self._horizon = env.spec.timestep_limit
        self._log_dir = log_dir
        self._force_reset = force_reset


    def step(self, action):
        s, reward, done, info = self.env.step(action)
        s = self._process_frame(s)
        s = s.astype(np.float32)
        s *= (1.0 / 255.0)
        self.buffer.append(s)
        s = np.array(self.buffer)
        s = np.transpose(s, (0, 3, 1, 2)).reshape((3*self.n_frames, 105, 80))
        s = self.premodel.predict(np.expand_dims(s, 0)).reshape((-1))
        return Step(s, reward, done, **info)

    def reset(self):
        self.buffer.clear()
        if self._force_reset and self.monitoring:
            recorder = self.env._monitor.stats_recorder
            if recorder is not None:
                recorder.done = True
        obs = self._process_frame(self.env.reset())
        frame = obs.astype(np.float32)
        frame *= (1.0 / 255.0)
        for _ in xrange(self.n_frames):
            self.buffer.append(frame)
        res = np.array(self.buffer)
        res = np.transpose(res, (0, 3, 1, 2)).reshape((self.n_frames*3, 105, 80))
        res = self.premodel.predict(np.expand_dims(res, 0)).reshape((-1))
        return res

    def _process_frame(self, frame):
        frame = cv2.resize(frame, (80, 105))
        return frame