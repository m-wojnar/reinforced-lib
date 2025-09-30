from argparse import ArgumentParser

import evosax.algorithms
import gymnasium as gym
import numpy as np
import optax
from chex import Array
from flax import linen as nn

from reinforced_lib import RLib
from reinforced_lib.agents.neuro import Evosax
from reinforced_lib.exts import GymnasiumVectorized
from reinforced_lib.logs import CsvLogger


class Network(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        action = 2 * nn.tanh(x)
        return action


def run(evo_alg: type, num_epochs: int, population_size: int, seed: int) -> None:
    """
    Run ``num_envs`` Pendulum Gymnasium environments in parallel using an evolutionary strategy to optimize the policy.

    Parameters
    ----------
    evo_alg : type
        Evolutionary strategy to use (from evosax).
    num_epochs : int
        Number of simulation steps to perform.
    population_size : int
        Number of parallel environments to use.
    seed : int
        Integer used as the random key.
    """

    evo_kwargs = {}
    evo_params = {}

    if isinstance(evo_alg, evosax.algorithms.CMA_ES):
        evo_params['std_init'] = 0.05
    elif isinstance(evo_alg, evosax.algorithms.SimpleES):
        evo_kwargs['optimizer'] = optax.adam(0.03)
        evo_params['std_init'] = 0.05
    elif isinstance(evo_alg, evosax.algorithms.SimulatedAnnealing):
        evo_kwargs['std_schedule'] = optax.constant_schedule(0.01)

    rl = RLib(
        agent_type=Evosax,
        agent_params={
            'network': Network(),
            'evo_strategy': evo_alg,
            'evo_strategy_kwargs': evo_kwargs,
            'evo_strategy_default_params': evo_params,
            'population_size': population_size
        },
        ext_type=GymnasiumVectorized,
        ext_params={'env_id': 'Pendulum-v1', 'num_envs': population_size},
        logger_types=CsvLogger,
        logger_params={'csv_path': f'pendulum-{evo_alg.__name__}-evo-{seed}.csv'}
    )

    def make_env():
        return gym.make('Pendulum-v1')

    env = gym.vector.SyncVectorEnv([make_env for _ in range(population_size)])

    for epoch in range(num_epochs):
        _, _ = env.reset(seed=seed + epoch)
        actions = env.action_space.sample()
        return_pop = np.zeros(population_size, dtype=float)

        for _ in range(env.envs[0].spec.max_episode_steps):
            env_states = env.step(np.asarray(actions))
            actions = rl.sample(*env_states)
            return_pop += env_states[1]

        rl.log('mean_return', return_pop.mean())
        rl.log('max_return', return_pop.max())
        rl.log('epoch', epoch + 1)


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--evo_alg', type=str, required=True)
    args.add_argument('--num_epochs', default=500, type=int)
    args.add_argument('--population_size', default=64, type=int)
    args.add_argument('--seed', default=42, type=int)

    args = args.parse_args()

    args = vars(args)
    args['evo_alg'] = getattr(evosax.algorithms, args['evo_alg'])
    run(**args)
