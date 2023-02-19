from argparse import ArgumentParser

import haiku as hk
from chex import Array

import gymnasium as gym
gym.logger.set_level(40)

from reinforced_lib import RLib
from reinforced_lib.agents.deep import QLearning
from reinforced_lib.logs import TensorboardLogger, SourceType
from reinforced_lib.exts import Gymnasium


@hk.transform_with_state
def q_network(x: Array) -> Array:
    return hk.nets.MLP([256, 2])(x)


def run(num_epochs: int, render_every: int, seed: int) -> None:
    """
    Run ``num_epochs`` cart-pole Gymnasium simulations with optional rendering.

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
        agent_type=QLearning,
        agent_params={'q_network': q_network},
        ext_type=Gymnasium,
        ext_params={'env_id': 'CartPole-v1'},
        loggers_type=TensorboardLogger,
        loggers_sources=[('reward', SourceType.METRIC), ('cumulative', SourceType.METRIC)]
    )

    for epoch in range(num_epochs):
        render = render_every is not None and epoch % render_every == 0
        env = gym.make('CartPole-v1', render_mode='human' if render else 'no')

        env_state, _ = env.reset(seed=seed + epoch)
        env_state = env.step(env.action_space.sample())
        terminal = False

        while not terminal:
            action = rl.sample(*env_state)
            env_state = env.step(action.item())

            terminal = env_state[2] or env_state[3]


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--num_epochs', default=2000, type=int)
    args.add_argument('--render_every', default=None, type=int)
    args.add_argument('--seed', default=42, type=int)

    args = args.parse_args()

    run(**(vars(args)))
