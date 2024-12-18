from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from chex import dataclass, PRNGKey, Scalar

from reinforced_lib.agents import AgentState, BaseAgent


@dataclass
class RandomSchedulerState(AgentState):
    r"""Random scheduler has no memory, thus the state is empty"""
    pass


class RandomScheduler(BaseAgent):
    r"""
        Random scheduler with MAB interface. This scheduler pics item randomly.

        Parameters
        ----------
        n_arms : int
            Number of bandit arms. :math:`N \in \mathbb{N}_{+}` .
    """

    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms

        self.init = jax.jit(self.init)
        self.update = jax.jit(self.update)
        self.sample = jax.jit(partial(self.sample, N=n_arms))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {'n_arms': gym.spaces.Box(1, jnp.inf, (1,), int), })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({'action': gym.spaces.Discrete(self.n_arms),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)})

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(self.n_arms)

    @staticmethod
    def init(key: PRNGKey) -> RandomSchedulerState:
        return RandomSchedulerState()

    @staticmethod
    def update(state: RandomSchedulerState, key: PRNGKey, action: int,
               reward: Scalar) -> RandomSchedulerState:
        return state

    @staticmethod
    def sample(state: RandomSchedulerState, key: PRNGKey, *args,
               **kwargs) -> int:
        N = kwargs.pop('N')
        a = jax.random.choice(key, N)
        return a
