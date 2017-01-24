
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite


#stub(globals())

#env = TfEnv(normalize(CartpoleEnv()))
#env = TfEnv(CartpoleEnv())
env = TfEnv(GymEnv('Skiing-v0'))
#env = gym.make('CartPole-v0')


policy = CategoricalMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
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