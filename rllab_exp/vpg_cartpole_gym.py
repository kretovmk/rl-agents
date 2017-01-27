
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite


#stub(globals())

env = TfEnv(GymEnv('CartPole-v0'))

policy = CategoricalMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32,)
)

#baseline = LinearFeatureBaseline(env_spec=env.spec)
baseline = ZeroBaseline(env_spec=env.spec)

algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=1000,
    max_path_length=1000,
    n_itr=100,
    discount=0.9,
    optimizer_args=dict(
        tf_optimizer_args=dict(
            learning_rate=0.001,
        )
    )
)

algo.train()


# run_experiment_lite(
#     algo.train(),
#     n_parallel=4,
#     seed=1,
#     #log_dir='exp/saved_models/'
# )