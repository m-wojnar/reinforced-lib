from argparse import ArgumentParser

import gymnasium as gym
import numpy as np
import optax
from chex import Array
from flax import linen as nn

from reinforced_lib import RLib
from reinforced_lib.agents.deep import PPODiscrete
from reinforced_lib.exts import GymnasiumVectorized
from reinforced_lib.logs import StdoutLogger, TensorboardLogger, WeightsAndBiasesLogger


class ActionNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        dense = lambda features, scale: nn.Dense(features, kernel_init=nn.initializers.orthogonal(scale))
        x = dense(64, scale=np.sqrt(2))(x)
        x = nn.tanh(x)
        x = dense(64, scale=np.sqrt(2))(x)
        x = nn.tanh(x)
        return dense(2, scale=0.01)(x)


class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        dense = lambda features, scale: nn.Dense(features, kernel_init=nn.initializers.orthogonal(scale))
        x = dense(64, scale=np.sqrt(2))(x)
        x = nn.tanh(x)
        x = dense(64, scale=np.sqrt(2))(x)
        x = nn.tanh(x)
        x = dense(1, scale=1)(x)
        return x.squeeze(axis=-1)


class PPOAgent(nn.Module):
    def setup(self) -> None:
        self.action_net = ActionNetwork()
        self.value_net = ValueNetwork()

    def __call__(self, x: Array) -> tuple[Array, Array]:
        logits = self.action_net(x)
        values = self.value_net(x)
        return logits, values


def run(num_steps: int, num_envs: int, seed: int) -> None:
    """
    Run ``num_steps`` cart-pole Gymnasium steps.

    Parameters
    ----------
    num_steps : int
        Number of simulation steps to perform.
    num_envs : int
        Number of parallel environments to use.
    seed : int
        Integer used as the random key.
    """

    rl = RLib(
        agent_type=PPODiscrete,
        agent_params={
            'network': PPOAgent(),
            'optimizer': optax.adam(1e-4, b1=0.9, eps=1e-5),
            'discount': 0.99,
            'lambda_gae': 0.9,
            'normalize_advantage': True,
            'clip_coef': 0.2,
            'clip_value': True,
            'clip_grad': 0.5,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'rollout_length': 32,
            'num_envs': num_envs,
            'batch_size': 512,
            'num_epochs': 4
        },
        ext_type=GymnasiumVectorized,
        ext_params={'env_id': 'CartPole-v1', 'num_envs': num_envs},
        logger_types=[StdoutLogger, TensorboardLogger, WeightsAndBiasesLogger]
    )

    def make_env():
        return gym.make('CartPole-v1', render_mode='no')

    step = 0

    while step < num_steps:
        env = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])

        _, _ = env.reset(seed=seed + step)
        actions = env.action_space.sample()

        terminal = np.array([False] * num_envs)
        max_epoch_len, min_epoch_len = 0, 0

        while not np.all(terminal):
            env_states = env.step(np.asarray(actions))
            actions = rl.sample(*env_states)

            terminal = terminal | env_states[2] | env_states[3]
            max_epoch_len += 1

            if not np.any(terminal):
                min_epoch_len += 1

        rl.log('max_epoch_len', max_epoch_len)
        rl.log('min_epoch_len', min_epoch_len)
        step += max_epoch_len * num_envs


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--num_steps', default=int(1e6), type=int)
    args.add_argument('--num_envs', default=64, type=int)
    args.add_argument('--seed', default=42, type=int)

    args = args.parse_args()

    run(**(vars(args)))
