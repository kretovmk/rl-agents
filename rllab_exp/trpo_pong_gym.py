
import tensorflow as tf

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from envs import GymEnvMod
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.core.network import ConvNetwork, MLP
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

# TODO: clarify with optimizers.. why finite differences?

#stub(globals())

ENV_NAME = 'Pong-v0'

env = TfEnv(GymEnvMod(env_name=ENV_NAME))

conv = ConvNetwork(name='CNN', input_shape=(105, 80, 1), output_dim=6,
                 conv_filters=(16, 32), conv_filter_sizes=(9, 5), conv_strides=(4, 2), conv_pads=('SAME', 'SAME'),
                 hidden_sizes=(256,), hidden_nonlinearity=tf.nn.relu, output_nonlinearity=tf.nn.softmax)

mlp = MLP(name='MLP', input_shape=(8400,), output_dim=6, hidden_sizes=(256,), hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=tf.nn.softmax)

policy = CategoricalMLPPolicy(
    name="policy",
    env_spec=env.spec,
    prob_network=conv,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)
#baseline = ZeroBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=30000,
    n_itr=1000,
    discount=1.0,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

algo.train()

# run_experiment_lite(
#     algo.train(),
#     n_parallel=4,
#     seed=1,
# )