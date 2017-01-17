
import numpy as np

from utils import make_list
from utils import floatX

# TODO: 1. add base clases, 2. add prioritized sampling

class ReplayMemory(object):
    """
    Replay memory class for off-policy algorithms. Allow n-step sampling.
    """
    def __init__(self, max_steps=100000, state_shape=(4,), state_dtype=floatX, num_continuous=0, seed=None):
        self.rng = np.random.RandomState(seed)
        self.max_steps = max_steps
        self.size = 0
        self.insert_ix = 0
        self.range = np.arange(max_steps)
        self.state_shape = state_shape
        self.state_dtype= state_dtype
        # initialize storing arrays
        self.states = self._init_batch(max_steps, state_shape, state_dtype)
        self.next_states = self._init_batch(max_steps, state_shape, state_dtype)
        self.rewards = np.zeros(max_steps, dtype=np.float32)
        self.terminals = np.zeros(max_steps, dtype=np.bool)
        if num_continuous > 0:
            self.actions = np.zeros((max_steps, num_continuous), dtype=np.float32)
            self.action_shape = (num_continuous,)
            self.action_dtype = floatX
        else:
            self.actions = np.zeros(max_steps, dtype=np.int32)
            self.action_shape = ()
            self.action_dtype = np.int32

    def reset(self):
        self.insert_ix = 0
        self.size = 0

    def add_sample(self, state, action, reward, next_state, terminal):
        self.actions[self.insert_ix] = action
        self.rewards[self.insert_ix] = reward
        self.terminals[self.insert_ix] = terminal
        self.states[self.insert_ix] = state
        self.next_states[self.insert_ix] = next_state
        self.insert_ix = (self.insert_ix + 1) % self.max_steps
        if self.size < self.max_steps:
            self.size += 1

    def get_random_batch(self, batch_size, n_steps):
        assert self.size > 2 * n_steps, 'Not enough samples in replay memory.'
        states = self._init_batch(batch_size, self.state_shape, self.state_dtype)
        next_states = self._init_batch(batch_size, self.state_shape, dtype=self.state_dtype)
        actions = self._init_batch(batch_size, self.action_shape, self.action_dtype)
        rewards = self._init_batch(batch_size, (n_steps,), dtype=np.float32)
        inv_rewards = self._init_batch(batch_size, (n_steps,), dtype=np.float32)
        terminals = self._init_batch(batch_size, shape=(1,), dtype='bool')
        inv_terminals = self._init_batch(batch_size, shape=(1,), dtype='bool')
        count = 0
        while count < batch_size:
            ix = np.random.randint(n_steps, self.size - n_steps)
            count_terminals = self.terminals.take(range(ix, ix + n_steps), axis=0)
            count_inv_terminals = self.terminals.take(range(ix - n_steps, ix), axis=0)
            if count_terminals.sum() > 0:
                ix_end = count_terminals.argmax(axis=0)
            else:
                ix_end = n_steps
            if count_inv_terminals.sum() > 0:
                ix_end_inv = count_inv_terminals[::-1].argmax(axis=0) + 1
            else:
                ix_end_inv = n_steps
            states[count] = self.states.take(ix, axis=0)
            next_states[count] = self.next_states.take(ix+ix_end, axis=0)
            actions[count] = self.actions.take(ix, axis=0)
            rewards[count, :ix_end] = self.rewards.take(range(ix, ix+ix_end), axis=0)
            inv_rewards[count, :ix_end_inv] = self.rewards.take(range(ix-ix_end_inv, ix), axis=0)
            terminals[count, :ix_end] = self.terminals.take(range(ix, ix+ix_end), axis=0)
            inv_terminals[count, :ix_end_inv] = self.terminals.take(range(ix-ix_end_inv, ix), axis=0)
            count += 1
        return states, actions, rewards, terminals, next_states, inv_rewards, inv_terminals

    def __len__(self):
        return self.size

    def _init_batch(self, size, shape=(), dtype=floatX):
        batch_shape = [size,] + make_list(shape)
        return np.zeros(batch_shape, dtype=dtype)
