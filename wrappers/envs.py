
import cv2
import numpy as np

from collections import deque
from gym.spaces.box import Box


def _process_frame(frame, rgb):
    frame = cv2.resize(frame, (80, 105))
    if not rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame

class AtariStackFrames(object):
    def __init__(self, env, n_frames=4, rgb=False):
        self.env = env
        self.rgb = rgb
        self.action_space = self.env.action_space
        self.observation_space = Box(0, 255, [n_frames*3, 105, 80])
        self.n_frames = n_frames
        self._checkpoint_buffer = []
        self.buffer = deque(maxlen=n_frames)

    def reset(self):
        self.buffer.clear()
        observation = self.env.reset()
        frame = _process_frame(observation, self.rgb)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        for _ in xrange(self.n_frames):
            self.buffer.append(frame)
        res = np.array(self.buffer)
        if self.rgb:
            res = np.transpose(res, (0, 3, 1, 2)).reshape((self.n_frames*3, 105, 80))
        return res

    def step(self, action):
        s, r, t, info = self.env.step(action)
        s = _process_frame(s, self.rgb)
        # process frame
        s = s.astype(np.float32)
        s *= (1.0 / 255.0)
        self.buffer.append(s)
        res = np.array(self.buffer)
        if self.rgb:
            res = np.transpose(res, (0, 3, 1, 2)).reshape((self.n_frames*3, 105, 80))
        return res, r, t, info


    def load_from_checkpoint(self):
        for frame in self._checkpoint_buffer:
            self.buffer.append(frame)
        del self._checkpoint_buffer[:]
        self.env.ale.loadState()

    def create_checkpoint(self):
        for frame in self.buffer:
            self._checkpoint_buffer.append(frame)
        self.env.ale.saveState()
