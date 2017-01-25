
import gym
import tensorflow as tf

import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from envs import GymEnvFrameProcessed, TfEnvFrameProcessed
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.core.network import ConvNetwork
from rllab.misc.instrument import stub, run_experiment_lite


#stub(globals())
ENV_NAME = 'Pong-v0'

#env = TfEnv(GymEnv(ENV_NAME))
#env = TfEnvFrameProcessed(GymEnvFrameProcessed(ENV_NAME))

conv = ConvNetwork(name='CNN', input_shape=(210, 160, 3), output_dim=6,
                 conv_filters=(16,), conv_filter_sizes=(1,), conv_strides=(4,), conv_pads=('SAME', 'SAME'),
                 hidden_sizes=(256,), hidden_nonlinearity=tf.nn.relu, output_nonlinearity=tf.nn.relu)

policy = CategoricalMLPPolicy(
    name="policy",
    env_spec=env.spec,
    prob_network=conv,
)

#baseline = LinearFeatureBaseline(env_spec=env.spec)
baseline = ZeroBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=10000,
    max_path_length=6100,
    n_itr=40,
    discount=1.0,
    step_size=0.01,
    # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

algo.train()

# run_experiment_lite(
#     algo.train(),
#     n_parallel=4,
#     seed=1,
# )