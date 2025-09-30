import time
from argparse import ArgumentParser

import gymnasium as gym
import numpy as np
import optax
from chex import Array
from flax import linen as nn

from reinforced_lib import RLib
from reinforced_lib.agents.deep import PPODiscrete
from reinforced_lib.exts import GymnasiumVectorized
from reinforced_lib.logs import CsvLogger


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


def run(time_limit: float, num_envs: int, seed: int) -> None:
    """
    Run ``num_envs`` CartPole Gymnasium environments in parallel using PPO to optimize the policy.
    The experiment runs for a maximum of ``time_limit`` seconds.

    Parameters
    ----------
    time_limit : float
        Maximum time (in seconds) to run the experiment.
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
            'batch_size': (num_envs * 32) // 4,
            'num_epochs': 4
        },
        ext_type=GymnasiumVectorized,
        ext_params={'env_id': 'CartPole-v1', 'num_envs': num_envs},
        logger_types=[CsvLogger],
        logger_params={'csv_path': f'cartpole-ppo-{num_envs}-envs-{seed}.csv'}
    )

    def make_env():
        return gym.make('CartPole-v1')

    env = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    _, _ = env.reset(seed=seed)

    actions = env.action_space.sample()
    return_0, step = 0, 0
    start_time = time.perf_counter()

    while time.perf_counter() - start_time < time_limit:
        env_states = env.step(np.asarray(actions))
        actions = rl.sample(*env_states)

        return_0 += env_states[1][0]
        step += num_envs

        if env_states[2][0] or env_states[3][0]:
            rl.log('return', return_0)
            rl.log('steps', step)
            rl.log('time', time.perf_counter() - start_time)
            return_0 = 0


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--time_limit', default=85, type=float)
    args.add_argument('--num_envs', default=64, type=int)
    args.add_argument('--seed', default=42, type=int)

    args = args.parse_args()

    run(**(vars(args)))
