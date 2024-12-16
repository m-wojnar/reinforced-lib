from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from chex import dataclass, Array, PRNGKey, Scalar

from reinforced_lib.agents import AgentState, BaseAgent


@dataclass
class RoundRobinState(AgentState):
    r"""
    Container for the state of the round-robin scheduler.

    Attributes
    ----------
    item : Array
        Scheduled item.
    """
    item: Array


class RoundRobinScheduler(BaseAgent):
    r"""
        Round-robin with MAB interface. This scheduler pics item sequentially.
        Sampling is deterministic, one must call ``update`` to change state.

        Parameters
        ----------
        n_arms : int
            Number of bandit arms. :math:`N \in \mathbb{N}_{+}` .
        starting_arm: int
            Initial arm to start sampling from.
    """

    def __init__(self, n_arms: int, starting_arm: int) -> None:
        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, item=starting_arm))
        self.update = jax.jit(partial(self.update, N=n_arms))
        self.sample = jax.jit(self.sample)

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
    def init(key: PRNGKey, item: int) -> RoundRobinState:
        return RoundRobinState(item=jnp.asarray(item))

    @staticmethod
    def update(state: RoundRobinState, key: PRNGKey, action: int,
               reward: Scalar, N: int) -> RoundRobinState:
        a = state.item + jnp.ones_like(state.item)
        a = jnp.mod(a, N)
        return RoundRobinState(item=a)

    @staticmethod
    def sample(state: RoundRobinState, key: PRNGKey, *args, **kwargs) -> int:
        return state.item
