
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import run_experiment_lite
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

def run_task(*_):
    from envs import GymEnvMod
    ENV_NAME = 'MsPacman-v0'
    env = GymEnvMod(env_name=ENV_NAME)

    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=()
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

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

run_experiment_lite(
    run_task,
    n_parallel=4,
    seed=1,
)
