
import tensorflow as tf
import numpy as np
import itertools
import os

from memory import ReplayMemory
from utils import EpisodeStats, copy_parameters


# TODO: concatenation of screens / states. how?

class DQNAgent(object):
    """
    Class for agent governed by DQN.
    """
    def __init__(self, sess,
                    env,
                    q_model,
                    target_model,
                    state_processor,
                    num_episodes,
                    experiments_folder,
                    replay_memory_size,
                    replay_memory_size_init,
                    upd_target_freq=10000,
                    gamma=0.99,
                    eps_start=1.0,
                    eps_end=0.1,
                    eps_decay_steps=500000,
                    batch_size=128,
                    record_video_freq=500):
        self.sess = sess
        self.q_model = q_model
        self.target_model = target_model
        self.replay_memory = replay_memory

    def choose_action(self, state, eps):
        if np.random.rand() < eps:
            action = np.random.randint(0, self.q_model.n_actions)
        else:
            action_probs = self.q_model.predict_q_values(self.sess, [state])
            action = np.argmax(action_probs)
        return action






def deep_q_learning(sess,
                    env,
                    q_model,
                    target_model,
                    state_processor,
                    num_episodes,
                    experiments_folder,
                    replay_memory_size,
                    replay_memory_size_init,
                    upd_target_freq=10000,
                    gamma=0.99,
                    eps_start=1.0,
                    eps_end=0.1,
                    eps_decay_steps=500000,
                    batch_size=128,
                    record_video_freq=500):



    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes),
                         episode_rewards=np.zeros(num_episodes))

    checkpoints_dir = os.path.join(experiments_folder, 'checkpoints')
    checkpoints_path = os.path.join(checkpoints_dir, 'model')
    monitor_path = os.path.join(checkpoints_dir, 'monitor')


    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    if latest_checkpoint:
        print 'Loading model checkpoint {} ..'.format(latest_checkpoint)
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(q_model.global_step)
    epsilons = np.linspace(eps_start, eps_end, eps_decay_steps)

    print 'Populating replay memory..'

    state = env.reset()
    state = state_processor.process(sess, state)  # TODO: why stack in Denny Britz's code? -- see line 334
    for i in xrange(replay_memory_size_init):
        eps = epsilons[min(total_t, eps_decay_steps - 1)]
        if np.random.rand() < eps:
            action = np.random.randint(0, q_model.n_actions)
        else:
            action_probs = q_model.predict_q_values(sess, [state])
            action = np.argmax(action_probs)

        next_state, reward, terminal, _ = env.step(action)
        next_state = state_processor.process(sess, next_state)
        replay_memory.add_sample(state, action, reward, terminal)
        if terminal:
            state = env.reset()
            state = state_processor.process(sess, state)
        else:
            state = next_state

    # TODO: fix eps schedule -- now doesn't work as it should
    # TODO: fix this.. doesn't work
    #env.monitor.start(monitor_path, resume=True, video_callable=lambda count: count % record_video_freq == 0)

    for i_episode in xrange(num_episodes):
        saver.save(tf.get_default_session(), checkpoints_path)

        state = env.reset()
        state = state_processor.process(sess, state)
        loss = None

        for t in itertools.count():
            eps = epsilons[min(total_t, eps_decay_steps - 1)]
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=eps, tag="epsilon")
            q_model.summary_writer.add_summary(episode_summary, total_t)

            if total_t % upd_target_freq == 0:
                copy_parameters(sess, q_model, target_model)
                print('Weights of target network were updated.')


            if np.random.rand() < eps:
                action = np.random.randint(0, q_model.n_actions)
            else:
                action_probs = q_model.predict_q_values(sess, [state])
                action = np.argmax(action_probs)

            next_state, reward, terminal, _ = env.step(action)
            next_state = state_processor.process(sess, next_state)
            replay_memory.add_sample(state, action, reward, terminal)


            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            batch = replay_memory.get_random_batch(batch_size=batch_size)
            states = batch['observations']
            actions = batch['actions']
            rewards = batch['rewards']
            terminals = batch['terminals']
            next_states = batch['next_observations']



            # double DQN
            q_values_next = q_model.predict_q_values(sess, next_states)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_model.predict_q_values(sess, next_states)
            targets = rewards + np.invert(terminals).astype(np.float32) \
                                * gamma * q_values_next_target[np.arange(batch_size), best_actions].reshape((-1, 1))
            loss = q_model.update_step(sess, states, targets.flatten(), actions)

            if terminal or t == 1000:
                print i_episode, t, eps, q_values_next_target[-1], sess.run(q_model.global_step)
                #print q_values_next_target, rewards, targets, terminal, terminals
                break

            state = next_state
            total_t += 1


        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward",
                                  tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length",
                                  tag="episode_length")
        q_model.summary_writer.add_summary(episode_summary, total_t)
        q_model.summary_writer.flush()


        # yield total_t, EpisodeStats(
        #     episode_lengths=stats.episode_lengths[:i_episode + 1],
        #     episode_rewards=stats.episode_rewards[:i_episode + 1])


    env.monitor.close()

    return stats

