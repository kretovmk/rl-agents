
import tensorflow as tf
import gym
import os

from dqn import QvalueEstimatorDense, deep_q_learning
from preprocessing import EmptyProcessor


if __name__ == '__main__':
    tf.reset_default_graph()
    env = gym.make('CartPole-v0')
    experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))
    global_step = tf.Variable(0, name='global_step', trainable=False)

    q_model = QvalueEstimatorDense(inp_shape=(4,), n_actions=2, scope="q", sum_dir=experiment_dir)
    target_model = QvalueEstimatorDense(inp_shape=(4,), n_actions=2, scope="target_q", sum_dir=experiment_dir)
    state_processor = EmptyProcessor()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        deep_q_learning(sess=sess,
                        env=env,
                        q_model=q_model,
                        target_model=target_model,
                        state_processor=state_processor,
                        num_episodes=10000,
                        experiments_folder=experiment_dir,
                        replay_memory_size=100000,
                        replay_memory_size_init=1000,
                        upd_target_freq=1000,
                        gamma=0.9,
                        eps_start=1.0,
                        eps_end=0.01,
                        eps_decay_steps=5000,
                        batch_size=64)