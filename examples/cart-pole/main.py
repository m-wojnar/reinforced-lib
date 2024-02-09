from argparse import ArgumentParser

import haiku as hk
import optax
from chex import Array

import gymnasium as gym
gym.logger.set_level(40)

from reinforced_lib import RLib
from reinforced_lib.agents.deep import QLearning
from reinforced_lib.exts import Gymnasium
from reinforced_lib.logs import StdoutLogger, TensorboardLogger, WeightsAndBiasesLogger


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
        agent_params={
            'q_network': q_network,
            'optimizer': optax.rmsprop(3e-4, decay=0.95, eps=1e-2),
            'discount': 0.95,
            'epsilon_decay': 0.9975
        },
        ext_type=Gymnasium,
        ext_params={'env_id': 'CartPole-v1'},
        logger_types=[StdoutLogger, TensorboardLogger, WeightsAndBiasesLogger]
    )

    for epoch in range(num_epochs):
        render = render_every is not None and epoch % render_every == 0
        env = gym.make('CartPole-v1', render_mode='human' if render else 'no')

        _, _ = env.reset(seed=seed + epoch)
        action = env.action_space.sample()

        terminal = False
        epoch_len = 0

        while not terminal:
            env_state = env.step(action.item())
            action = rl.sample(*env_state)

            terminal = env_state[2] or env_state[3]
            epoch_len += 1

        rl.log('epoch_len', epoch_len)


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--num_epochs', default=300, type=int)
    args.add_argument('--render_every', default=None, type=int)
    args.add_argument('--seed', default=42, type=int)

    args = args.parse_args()

    run(**(vars(args)))
