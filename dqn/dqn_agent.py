
import tensorflow as tf
import numpy as np
import logging
import math

from utils import copy_parameters

logger = logging.getLogger('__main__')

# TODO: concatenation of screens / states. how?

class RandomAgent(object):
    """
    Random uniform sampling from discrete action space.
    """
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def choose_action(self, state):
        return np.random.randint(0, self.n_actions)


class DQNAgent(object):
    """
    Class for agent governed by DQN.
    """
    def __init__(self, sess,
                       env,
                       double_dqn,
                       q_model,
                       target_model,
                       replay_memory,
                       state_processor,
                       upd_target_freq=10000,
                       gamma=0.99,
                       eps_start=1.0,
                       eps_end=0.1,
                       eps_lambda=0.001,
                       batch_size=128):
        self.sess = sess
        self.env = env
        self.double_dqn = double_dqn
        self.q_model = q_model
        self.target_model = target_model
        self.replay_memory = replay_memory
        self.state_processor = state_processor
        self.upd_target_freq = upd_target_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_lambda = eps_lambda
        self.gamma = gamma
        self.batch_size = batch_size

    def choose_action(self, state, eps):
        if np.random.rand() < eps:
            action = np.random.randint(0, self.q_model.n_actions)
        else:
            action_probs = self.q_model.predict_q_values(self.sess, [state])
            action = np.argmax(action_probs)
        return action

    def _process_and_stack(self, state, prev_state=None):
        """
        Make preprocessing and concatenate states if needed.
        """
        state = self.state_processor.process(self.sess, state)
        if self.replay_memory.concat_observations == 1:
            return np.stack([state], axis=0)
        else:
            if not prev_state:
                return np.stack([state] * self.replay_memory.concat_observations, axis=0)
            else:
                return np.append(state[1:], np.expand_dims(state, 0), axis=0)

    def _decrease_eps(self, t):
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.eps_lambda * t)

    def _train_step(self):
        pass

    def fill_replay_memory(self, exploration_agent, steps):
        """
        Fill in replay memory for specified number of steps.
        """
        logger.info('Populating replay memory..')
        state = self.env.reset()
        state = self._process_and_stack(state)
        for i in xrange(steps):
            action = exploration_agent(state)
            next_state, reward, terminal, _ = self.env.step(action)
            next_state = self._process_and_stack(next_state, state)
            self.replay_memory.add_sample(state, action, reward, terminal)
            if terminal:
                state = self.env.reset()
                state = self._process_and_stack(state)
            else:
                state = next_state

    def run_episode(self, test=False, max_steps=1000):
        """
        Run single episode.
        """
        total_t = self.sess.run(self.q_model.global_step)
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=self.eps, tag="epsilon")
        self.q_model.summary_writer.add_summary(episode_summary, total_t)
        total_reward = 0.
        state = self.env.reset()
        state = self._process_and_stack(state)
        step = 0
        while True:
            if test:
                action = self.choose_action(state, eps=0.)
            else:
                action = self.choose_action(state, eps=self.eps)
            next_state, reward, terminal, _ = self.env.step(action)
            total_reward += reward
            next_state = self.state_processor.process(self.sess, next_state)
            self.replay_memory.add_sample(state, action, reward, terminal)

            # training
            if not test:
                self.replay_memory.add_sample(state, action, reward, terminal)
                total_t = self.sess.run(self.q_model.global_step)
                self._decrease_eps(total_t)
                if total_t % self.upd_target_freq == 0:
                    copy_parameters(self.sess, self.q_model, self.target_model)
                    logging.info('Training step: {}, weights of target network were updated.'.format(total_t))
                batch = self.replay_memory.get_random_batch(batch_size=self.batch_size)
                states = batch['observations']
                actions = batch['actions']
                rewards = batch['rewards']
                terminals = batch['terminals']
                next_states = batch['next_observations']

                if self.double_dqn:
                    q_values_next = self.q_model.predict_q_values(self.sess, next_states)
                    best_actions = np.argmax(q_values_next, axis=1)
                else:
                    q_values_next = self.target_model.predict_q_values(self.sess, next_states)
                    best_actions = np.argmax(q_values_next, axis=1)

                q_values_next_target = self.target_model.predict_q_values(self.sess, next_states)
                targets = rewards + np.invert(terminals).astype(np.float32) \
                        * self.gamma * q_values_next_target[np.arange(self.batch_size), best_actions].reshape((-1, 1))
                loss = self.q_model.update_step(self.sess, states, targets.flatten(), actions)

            if terminal or step == max_steps:
                return total_reward

            state = next_state
            step += 1
            total_t += 1

            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=total_reward, node_name="episode_reward",
                                      tag="episode_reward")
            self.q_model.summary_writer.add_summary(episode_summary, total_t)
            self.q_model.summary_writer.flush()
