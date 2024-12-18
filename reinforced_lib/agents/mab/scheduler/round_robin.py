from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from chex import dataclass, Numeric, PRNGKey

from reinforced_lib.agents import AgentState, BaseAgent


@dataclass
class RoundRobinSchedulerState(AgentState):
    r"""
    Container for the state of the round-robin scheduler.

    Attributes
    ----------
    item : Numeric
        Scheduled item.
    """

    item: Numeric


class RoundRobinScheduler(BaseAgent):
    r"""
    Round-robin with MAB interface. This scheduler pics item sequentially.
    Sampling is deterministic, one must call ``update`` to change state.

    Parameters
    ----------
    n_arms : int
        Number of items to choose from. :math:`N \in \mathbb{N}_{+}`.
    initial_item: int, default=0
        Initial item to start sampling from.
    """

    def __init__(self, n_arms: int, initial_item: int = 0) -> None:
        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, item=initial_item))
        self.update = jax.jit(partial(self.update, n_arms=n_arms))
        self.sample = jax.jit(self.sample)

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), int)
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(self.n_arms)

    @staticmethod
    def init(key: PRNGKey, item: Numeric) -> RoundRobinSchedulerState:
        return RoundRobinSchedulerState(item=jnp.asarray(item))

    @staticmethod
    def update(state: RoundRobinSchedulerState, key: PRNGKey, n_arms: int) -> RoundRobinSchedulerState:
        return RoundRobinSchedulerState(item=(state.item + 1) % n_arms)

    @staticmethod
    def sample(state: RoundRobinSchedulerState, key: PRNGKey) -> Numeric:
        return state.item
