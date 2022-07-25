from functools import partial
from typing import Tuple

import gym.spaces
import jax
import jax.numpy as jnp
from chex import dataclass, Array, PRNGKey

from reinforced_lib.agents.base_agent import BaseAgent, AgentState


@dataclass
class EGreedyState(AgentState):
    """
    Container for the state of the e-greedy agent.

    Attributes
    ----------
    e : float
        The experiment rate
    Q : array_like
        Quality values for each arm
    N : array_like
        Number of tries for each arm
    """
    e: jnp.float32
    Q: Array
    N: Array


class EGreedy(BaseAgent):
    """
    Epsilon-greedy agent with optional optimistic start behaviour.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms.
    e : float
        Experiment rate (epsilon).
    optimistic_start : float, default=0.0
        If non-zero than it is interpreted as the optimistic start to encourage
        exploration, by default 0.0.
    """

    def __init__(self, n_arms: jnp.int32, e: jnp.float32, optimistic_start: jnp.float32 = 0.0) -> None:
        self.n_arms = n_arms
        self.e = e
        self.optimistic_start = optimistic_start

        self.init = jax.jit(partial(self.init, n_arms=n_arms, e=e, optimistic_start=optimistic_start))
        self.update = jax.jit(partial(self.update))
        self.sample = jax.jit(partial(self.sample))
    
    @staticmethod
    def parameters_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'e': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32),
            'optimistic_start': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32)
        })
    
    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_arms),
            'reward': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32)
        })
    
    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})
    
    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(self.n_arms)

    @staticmethod
    def init(
            key: PRNGKey,
            n_arms: jnp.int32,
            e: jnp.float32,
            optimistic_start: jnp.float32
    ) -> EGreedyState:
        """
        Creates and initializes instance of the e-greedy agent.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of contextual bandit arms.
        e : float
            Experiment rate (epsilon).
        optimistic_start : float
            If non zero than it is interpreted as the optimistic start to encourage exploration.

        Returns
        -------
        state : EGreedyState
            Initial state of the e-greedy agent.
        """

        return EGreedyState(
            e=e,
            Q=(optimistic_start * jnp.ones(n_arms)),
            N=jnp.ones(n_arms, dtype=jnp.int32)
        )

    @staticmethod
    def update(
        state: EGreedyState,
        key: PRNGKey,
        action: jnp.int32,
        reward: jnp.float32,
    ) -> EGreedyState:
        """
        Updates the state of the agent after performing some action and receiving a reward.

        Parameters
        ----------
        state : EGreedyState
            Current state of agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : jnp.int32
            Previously selected action.
        reward : jnp.float32
            Reward as a result of previous action.

        Returns
        -------
        EGreedyState
            Updated agent state.
        """
        
        return EGreedyState(
            e=state.e,
            Q=state.Q.at[action].add((1.0 / state.N[action]) * (reward - state.Q[action])),
            N=state.N.at[action].add(1)
        )

    @staticmethod
    def sample(
        state: EGreedyState,
        key: PRNGKey
    ) -> Tuple[EGreedyState, jnp.int32]:
        """
        Selects next action based on current agent state.

        Parameters
        ----------
        state : EGreedyState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.

        Returns
        -------
        tuple[EGreedyState, jnp.int32]
            Tuple containing updated agent state and selected action.
        """

        return jax.lax.cond(
            jax.random.uniform(key) < state.e,
            lambda: (state, jax.random.choice(key, state.Q.size)),
            lambda: (state, jnp.argmax(state.Q))
        )
