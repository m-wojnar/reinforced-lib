from functools import partial
from typing import Tuple

import gym.spaces
import jax
import jax.numpy as jnp
from chex import dataclass, Array, Scalar, PRNGKey

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class EGreedyState(AgentState):
    """
    Container for the state of the e-greedy agent.

    Attributes
    ----------
    Q : array_like
        Quality values for each arm
    N : array_like
        Number of tries for each arm
    """

    Q: Array
    N: Array


class EGreedy(BaseAgent):
    """
    Epsilon-greedy agent with optional optimistic start behaviour and exponential recency-weighted average update.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms.
    e : float
        Experiment rate (epsilon).
    optimistic_start : float, default=0.0
        If non-zero than it is interpreted as the optimistic start to encourage exploration.
    alpha : float, default=0.0
        If non-zero than exponential recency-weighted average is used to update Q values. ``alpha`` must be in [0, 1).
    """

    def __init__(
            self,
            n_arms: jnp.int32,
            e: Scalar,
            optimistic_start: Scalar = 0.0,
            alpha: Scalar = 0.0
    ) -> None:
        assert 0 <= alpha <= 1

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=n_arms, optimistic_start=optimistic_start))
        self.update = jax.jit(partial(self.update, alpha=alpha))
        self.sample = jax.jit(partial(self.sample, e=e))

    @staticmethod
    def parameters_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'e': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32),
            'optimistic_start': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32),
            'alpha': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32)
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_arms),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), jnp.float32)
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
            optimistic_start: Scalar
    ) -> EGreedyState:
        """
        Creates and initializes instance of the e-greedy agent.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of contextual bandit arms.
        optimistic_start : float
            Initial estimated action value.

        Returns
        -------
        state : EGreedyState
            Initial state of the e-greedy agent.
        """

        return EGreedyState(
            Q=(optimistic_start * jnp.ones(n_arms)),
            N=jnp.ones(n_arms, dtype=jnp.int32)
        )

    @staticmethod
    def update(
        state: EGreedyState,
        key: PRNGKey,
        action: jnp.int32,
        reward: Scalar,
        alpha: Scalar
    ) -> EGreedyState:
        """
        Updates the state of the agent after performing some action and receiving a reward.

        Parameters
        ----------
        state : EGreedyState
            Current state of agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        reward : float
            Reward as a result of previous action.
        alpha : float
            Exponential recency-weighted average factor (used when ``alpha > 0``).

        Returns
        -------
        EGreedyState
            Updated agent state.
        """

        def classic_update(operands: Tuple) -> EGreedyState:
            state, action, reward, alpha = operands
            return EGreedyState(
                Q=state.Q.at[action].add((reward - state.Q[action]) / state.N[action]),
                N=state.N.at[action].add(1)
            )

        def erwa_update(operands: Tuple) -> EGreedyState:
            state, action, reward, alpha = operands
            return EGreedyState(
                Q=state.Q.at[action].add(alpha * (reward - state.Q[action])),
                N=state.N.at[action].add(1)
            )

        return jax.lax.cond(alpha == 0, classic_update, erwa_update, (state, action, reward, alpha))

    @staticmethod
    def sample(
        state: EGreedyState,
        key: PRNGKey,
        e: Scalar
    ) -> Tuple[EGreedyState, jnp.int32]:
        """
        Selects next action based on current agent state.

        Parameters
        ----------
        state : EGreedyState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        e : float
            Experiment rate (epsilon).

        Returns
        -------
        tuple[EGreedyState, jnp.int32]
            Tuple containing updated agent state and selected action.
        """

        epsilon_key, choice_key = jax.random.split(key)

        return jax.lax.cond(
            jax.random.uniform(epsilon_key) < e,
            lambda: (state, jax.random.choice(choice_key, state.Q.size)),
            lambda: (state, jnp.argmax(state.Q))
        )
