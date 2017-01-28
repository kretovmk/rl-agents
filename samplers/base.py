
import numpy as np
import itertools
import logging

from utils.math import discount_rewards

logger = logging.getLogger('__main__')

# TODO: add multiprocessing

class SamplerBase(object):

    def __init__(self, sess, env, policy, state_processor, max_steps):
        self.sess = sess
        self.env = env
        self.policy = policy
        self.state_processor = state_processor
        self.max_steps = max_steps

    def run_episode(self, gamma, sample=True):
        states = []
        actions = []
        rewards = []
        state = self.env.reset()
        state = self.state_processor.process(self.sess, state)
        for i in xrange(self.max_steps):
            prob_actions = self.policy.predict_x(self.sess, state)
            if sample:
                action = np.random.choice(np.arange(len(prob_actions)), p=prob_actions)
            else:
                action = np.argmax(prob_actions)
            next_state, reward, terminal, _ = self.env.step(action)
            next_state = self.state_processor.process(self.sess, next_state)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if terminal:
                break
        returns = discount_rewards(np.array(rewards), gamma)
        return dict(states=np.array(states),
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    returns=returns)

    def collect_samples(self, gamma, batch_size, sample=True):
        paths = []
        prev_len = 0
        for t in itertools.count():
            paths.append(self.run_episode(gamma, sample))
            cur_len = sum([len(x['actions']) for x in paths])
            if cur_len >= batch_size:
                if t == 0:
                    logger.info('WARNING: just one truncated trajectory for batch. Consider increasing batch size.')
                ix = cur_len - batch_size
                for k, v in paths[-1].iteritems():
                    paths[-1][k] = v[:-ix]
                break
            prev_len = cur_len
        states = np.concatenate([x['states'] for x in paths])
        actions = np.concatenate([x['actions'] for x in paths])
        returns = np.concatenate([x['returns'] for x in paths])
        return dict(states=states, actions=actions, returns=returns)

    def test_agent(self, sample=False):
        res = self.run_episode(gamma=1., sample=sample)
        total_reward = res['returns'][0]
        episode_length = len(res['returns'])
        return total_reward, episode_length