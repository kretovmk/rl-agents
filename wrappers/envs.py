
import cv2
import numpy as np

from collections import deque
from gym.spaces.box import Box


def _process_frame(frame):
    frame = cv2.resize(frame, (80, 105))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame

class AtariStackFrames(object):
    def __init__(self, env, n_frames=4):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = Box(0, 255, [n_frames, 105, 80])
        self.n_frames = n_frames
        self._checkpoint_buffer = []
        self.buffer = deque(maxlen=n_frames)

    def reset(self):
        self.buffer.clear()
        observation = self.env.reset()
        frame = _process_frame(observation)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        for _ in xrange(self.n_frames):
            self.buffer.append(frame)
        return np.array(self.buffer)

    def step(self, action):
        s, r, t, info = self.env.step(action)
        s = _process_frame(s)
        # process frame
        s = s.astype(np.float32)
        s *= (1.0 / 255.0)
        self.buffer.append(s)
        return np.array(self.buffer), r, t, info

    def load_from_checkpoint(self):
        for frame in self._checkpoint_buffer:
            self.buffer.append(frame)
        del self._checkpoint_buffer[:]
        self.env.ale.loadState()

    def create_checkpoint(self):
        for frame in self.buffer:
            self._checkpoint_buffer.append(frame)
        self.env.ale.saveState()
