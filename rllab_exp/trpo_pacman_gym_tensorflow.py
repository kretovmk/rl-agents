
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

ENV_NAME = 'MsPacman-v0'

env = TfEnv(GymEnvMod(env_name=ENV_NAME))

policy = CategoricalMLPPolicy(
    name="policy",
    env_spec=env.spec,
    hidden_sizes=()
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
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(subsample_factor=0.1)
)

algo.train()

# run_experiment_lite(
#     algo.train(),
#     n_parallel=4,
#     seed=1,
# )