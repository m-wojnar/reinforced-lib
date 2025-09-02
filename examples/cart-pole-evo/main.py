from argparse import ArgumentParser

import gymnasium as gym
import jax
import numpy as np
from chex import Array
from evosax.algorithms import PGPE
from flax import linen as nn

from reinforced_lib import RLib
from reinforced_lib.agents.neuro import Evosax
from reinforced_lib.exts import GymnasiumVectorized
from reinforced_lib.logs import StdoutLogger, TensorboardLogger, WeightsAndBiasesLogger


class Network(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        logits = nn.Dense(2)(x)
        action = jax.random.categorical(self.make_rng('rlib'), logits)
        return action


def run(num_epochs: int, population_size: int, seed: int) -> None:
    """
    Run ``num_epochs`` cart-pole Gymnasium environments in parallel using an evolutionary strategy to optimize the policy.

    Parameters
    ----------
    num_epochs : int
        Number of simulation steps to perform.
    population_size : int
        Number of parallel environments to use.
    seed : int
        Integer used as the random key.
    """

    rl = RLib(
        agent_type=Evosax,
        agent_params={
            'network': Network(),
            'evo_strategy': PGPE,
            'evo_strategy_default_params': {'std_init': 0.1},
            'population_size': population_size
        },
        ext_type=GymnasiumVectorized,
        ext_params={'env_id': 'CartPole-v1', 'num_envs': population_size},
        logger_types=[StdoutLogger, TensorboardLogger, WeightsAndBiasesLogger]
    )

    def make_env():
        return gym.make('CartPole-v1', render_mode='no')

    for step in range(num_epochs):
        env = gym.vector.SyncVectorEnv([make_env for _ in range(population_size)])

        _, _ = env.reset(seed=seed + step)
        actions = env.action_space.sample()

        terminal = np.array([False] * population_size)
        max_epoch_len = 0

        while not np.all(terminal):
            env_states = env.step(np.asarray(actions))
            actions = rl.sample(*env_states)

            terminal = terminal | env_states[2] | env_states[3]
            max_epoch_len += 1

        rl.log('max_epoch_len', max_epoch_len)


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--num_epochs', default=300, type=int)
    args.add_argument('--population_size', default=64, type=int)
    args.add_argument('--seed', default=42, type=int)

    args = args.parse_args()

    run(**(vars(args)))
