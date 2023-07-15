from argparse import ArgumentParser

import haiku as hk
import jax.numpy as jnp
import optax
from chex import Array

import gymnasium as gym
gym.logger.set_level(40)

from reinforced_lib import RLib
from reinforced_lib.agents.deep import DDPG
from reinforced_lib.exts import Gymnasium
from reinforced_lib.logs import StdoutLogger, TensorboardLogger


@hk.transform_with_state
def q_network(s: Array, a: Array) -> Array:
    s = hk.nets.MLP([16, 32, 8])(s)
    x = jnp.concatenate([s, a], axis=-1)
    return hk.nets.MLP([256, 256, 1])(x)


@hk.transform_with_state
def a_network(s: Array) -> Array:
    return hk.nets.MLP([256, 256, 1])(s)


def run(num_epochs: int, render_every: int, seed: int) -> None:
    """
    Run ``num_epochs`` pendulum Gymnasium simulations with optional rendering.

    Parameters
    ----------
    num_epochs : int
        Number of simulation epochs to perform.
    render_every : int, optional
        Renders environment every ``auto_checkpoint`` steps. If ``None``, the rendering is disabled.
    seed : int
        Integer used as the random key.
    """

    rl = RLib(
        agent_type=DDPG,
        agent_params={
            'q_network': q_network,
            'a_network': a_network,
            'q_optimizer': optax.adam(2e-3),
            'a_optimizer': optax.adam(1e-3),
            'experience_replay_buffer_size': 50000,
            'experience_replay_batch_size': 64,
            'discount': 0.99,
            'noise': 0.2,
            'noise_decay': 1.0,
            'tau': 0.005,
        },
        ext_type=Gymnasium,
        ext_params={'env_id': 'Pendulum-v1'},
        logger_types=[StdoutLogger, TensorboardLogger]
    )

    for epoch in range(num_epochs):
        render = render_every is not None and epoch % render_every == 0
        env = gym.make('Pendulum-v1', render_mode='human' if render else 'no')

        _, _ = env.reset(seed=seed + epoch)
        action = env.action_space.sample()

        terminal = False
        rewards_sum = 0

        while not terminal:
            env_state = env.step(action)
            action = rl.sample(*env_state)

            terminal = env_state[2] or env_state[3]
            rewards_sum += env_state[1]

        rl.log('rewards_sum', rewards_sum)


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--num_epochs', default=300, type=int)
    args.add_argument('--render_every', default=None, type=int)
    args.add_argument('--seed', default=42, type=int)

    args = args.parse_args()

    run(**(vars(args)))
