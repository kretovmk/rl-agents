
import numpy as np

from utils import make_list


# TODO: add n_steps, prioritized sampling

class ReplayMemory(object):
    """
    A utility class for experience replay.
    The code is adapted from https://github.com/openai/rllab
    """

    def __init__(self,
                 observation_shape=(4,),
                 action_dim=1,
                 max_steps=1000000,
                 observation_dtype=np.float32,
                 action_dtype=np.int32,
                 concat_observations=False,
                 concat_length=1,
                 rng=None):
        """Construct a ReplayPool.
        Arguments:
            observation_shape - tuple indicating the shape of the observation
            action_dim - dimension of the action
            size - capacity of the replay pool
            observation_dtype - ...
            action_dtype - ...
            concat_observations - whether to concat the past few observations
            as a single one, so as to ensure the Markov property
            concat_length - length of the concatenation
        """
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.observations = np.zeros((max_steps,) + observation_shape, dtype=observation_dtype)
        self.actions = np.zeros((max_steps, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((max_steps, 1), dtype=np.float32)
        self.terminals = np.zeros((max_steps, 1), dtype='bool')
        self.concat_observations = concat_observations
        self.concat_length = concat_length
        self.observation_dtype = observation_dtype
        self.action_dtype = action_dtype
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()

        if not concat_observations:
            assert concat_length == 1, \
                "Concat_length must be set to 1 if not concatenating observations."
        self.bottom = 0
        self.top = 0
        self.size = 0

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in random_batch's check.
        return max(0, self.size - self.concat_length)

    def add_sample(self, observation, action, reward, terminal, extra=None):
        """Add a time step record.
        Arguments:
            observation -- current or observation
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended after this
            extra -- for example time step or replay priority
        """
        self.observations[self.top] = observation
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminals[self.top] = terminal
        if extra is not None:
            if self.extras is None:
                assert self.size == 0, "Extra data in replay memory must be consistent with other data."
                self.extras = np.zeros((self.max_steps,) + extra.shape, dtype=extra.dtype)
            self.extras[self.top] = extra
        else:
            assert self.extras is None

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def last_concat_state(self):
        """
        Return the most recent sample (concatenated observations if needed).
        """
        if self.concat_observations:
            indexes = np.arange(self.top - self.concat_length, self.top)
            return self.observations.take(indexes, axis=0, mode='wrap')
        else:
            return self.observations[self.top - 1]

    def concat_state(self, state):
        """Return a concatenated state, using the last concat_length -
        1, plus state.
        """
        if self.concat_observations:
            indexes = np.arange(self.top - self.concat_length + 1, self.top)
            concat_state = np.empty((self.concat_length,) + self.observation_shape, dtype=self.observation_dtype)
            concat_state[0: self.concat_length - 1] = self.observations.take(indexes, axis=0, mode='wrap')
            concat_state[-1] = state
            return concat_state
        else:
            return state

    def random_batch(self, batch_size=32, index=None):
        """
        Return corresponding observations, actions, rewards, terminal status,
        and next_observations for batch_size randomly chosen state transitions.
        """

        observations = np.zeros(
            (batch_size, self.concat_length) + self.observation_shape,
            dtype=self.observation_dtype
        )
        actions = np.zeros(
            (batch_size, self.action_dim),
            dtype=self.action_dtype
        )
        rewards = np.zeros((batch_size, 1), dtype=np.float32)
        terminals = np.zeros((batch_size, 1), dtype='bool')
        if self.extras is not None:
            extras = np.zeros(
                (batch_size,) + self.extras.shape[1:],
                dtype=self.extras.dtype
            )
            next_extras = np.zeros(
                (batch_size,) + self.extras.shape[1:],
                dtype=self.extras.dtype
            )
        else:
            extras = None
            next_extras = None
        next_observations = np.zeros(
            (batch_size, self.concat_length) + self.observation_shape,
            dtype=self.observation_dtype
        )
        next_actions = np.zeros(
            (batch_size, self.action_dim),
            dtype=self.action_dtype
        )

        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            if not index:
                index = self.rng.randint(
                    self.bottom,
                    self.bottom + self.size - self.concat_length
                )

            initial_indices = np.arange(index, index + self.concat_length)
            transition_indices = initial_indices + 1
            end_index = index + self.concat_length - 1

            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but the last frame may be
            # terminal. If the last frame of the initial state is
            # terminal, then the last frame of the transitioned state
            # will actually be the first frame of a new episode, which
            # the Q learner recognizes and handles correctly during
            # training by zeroing the discounted future reward estimate.
            if np.any(self.terminals.take(initial_indices[0:-1], mode='wrap')):
                continue
            # do not pick samples which terminated because of horizon
            # if np.any(self.horizon_terminals.take(initial_indices[0:-1],
            #    mode='wrap')) or self.horizon_terminals[end_index]:
            #    continue

            # Add the state transition to the response.
            observations[count] = self.observations.take(initial_indices, axis=0, mode='wrap')
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminals[count] = self.terminals.take(end_index, mode='wrap')
            if self.extras is not None:
                extras[count] = self.extras.take(end_index, axis=0, mode='wrap')
                next_extras[count] = self.extras.take(transition_indices, axis=0, mode='wrap')
            next_observations[count] = self.observations.take(transition_indices, axis=0, mode='wrap')
            next_actions[count] = self.actions.take(transition_indices, axis=0, mode='wrap')

            count += 1

        if not self.concat_observations:
            # If we're not concatenating observations, we should squeeze the
            # second dimension in observations and next_observations
            observations = np.squeeze(observations, axis=1)
            next_observations = np.squeeze(next_observations, axis=1)

        return dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_actions=next_actions,
            terminals=terminals,
            extras=extras,
            next_extras=next_extras,
            index=np.array([index]),
        )

    def random_batch_n_step(self, batch_size=32, n_steps=1, backward=False):
        batch_forward = []
        batch_backward = []

        for i in xrange(batch_size):
            start_transition = self.random_batch(batch_size=1)
            start_index = start_transition['index'][0]
            n_step_forward = start_transition

            # forward pass
            for j in xrange(1, n_steps):
                transition = start_transition
                if not transition['terminals'][0, 0]:
                    ix = transition['index'][0]
                    transition = self.random_batch(batch_size=1, index=ix+1)
                    for k, v in transition.iteritems():
                        n_step_forward[k] = np.concatenate((v, n_step_forward[k]), axis=0)
                else:
                    break
            batch_forward.append(n_step_forward)

            # backward pass
            if backward:
                transition = self.random_batch(batch_size=1, index=start_index-1)
                n_step_backward = transition
                for j in xrange(1, n_steps):
                    if not transition['terminals'][0, 0]:
                        ix = transition['index'][0]
                        transition = self.random_batch(batch_size=1, index=ix-1)
                        for k, v in transition.iteritems():
                            n_step_backward[k] = np.concatenate((v, n_step_backward[k]), axis=0)
                    else:
                        break
                batch_backward.append(n_step_backward)

        return batch_forward, batch_backward
