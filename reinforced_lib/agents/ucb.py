from functools import partial
from typing import Tuple

import gym.spaces
import jax
import jax.numpy as jnp
from chex import dataclass, Array, Scalar, PRNGKey

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class UCBState(AgentState):
    """
    Container for the state of the UCB agent.

    Attributes
    ----------
    Q : array_like
        Quality values for each arm
    N : array_like
        Number of tries for each arm
    """

    Q: Array
    N: Array


class UCB(BaseAgent):
    """
    UCB agent with optional exponential recency-weighted average update.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms.
    c : float
        Degree of exploration.
    alpha : float, default=0.0
        If non-zero than exponential recency-weighted average is used to update Q values. ``alpha`` must be in [0, 1).
    """

    def __init__(
            self,
            n_arms: jnp.int32,
            c: Scalar,
            alpha: Scalar = 0.0
    ) -> None:
        assert c >= 0
        assert 0 <= alpha <= 1

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=n_arms))
        self.update = jax.jit(partial(self.update, alpha=alpha))
        self.sample = jax.jit(partial(self.sample, c=c))

    @staticmethod
    def parameters_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'c': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32),
            'alpha': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32)
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_arms),
            'reward': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32)
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'time': gym.spaces.Box(0.0, jnp.inf, (1,))
        })

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(self.n_arms)

    @staticmethod
    def init(
            key: PRNGKey,
            n_arms: jnp.int32
    ) -> UCBState:
        """
        Creates and initializes instance of the UCB agent.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of contextual bandit arms.

        Returns
        -------
        state : UCBState
            Initial state of the UCB agent.
        """

        return UCBState(
            Q=jnp.zeros(n_arms),
            N=jnp.ones(n_arms, dtype=jnp.int32)
        )

    @staticmethod
    def update(
        state: UCBState,
        key: PRNGKey,
        action: jnp.int32,
        reward: Scalar,
        alpha: Scalar
    ) -> UCBState:
        """
        Updates the state of the agent after performing some action and receiving a reward.

        Parameters
        ----------
        state : UCBState
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
        UCBState
            Updated agent state.
        """

        def classic_update(operands: Tuple) -> UCBState:
            state, action, reward, alpha = operands
            return UCBState(
                Q=state.Q.at[action].add((1.0 / state.N[action]) * (reward - state.Q[action])),
                N=state.N.at[action].add(1)
            )

        def erwa_update(operands: Tuple) -> UCBState:
            state, action, reward, alpha = operands
            return UCBState(
                Q=state.Q.at[action].add(alpha * (reward - state.Q[action])),
                N=state.N.at[action].add(1)
            )

        return jax.lax.cond(alpha == 0, classic_update, erwa_update, (state, action, reward, alpha))

    @staticmethod
    def sample(
        state: UCBState,
        key: PRNGKey,
        time: Scalar,
        c: Scalar
    ) -> Tuple[UCBState, jnp.int32]:
        """
        Selects next action based on current agent state.

        Parameters
        ----------
        state : UCBState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        time : float
            Current time.
        c : float
            Degree of exploration.

        Returns
        -------
        tuple[UCBState, jnp.int32]
            Tuple containing updated agent state and selected action.
        """

        return state, jnp.argmax(state.Q + c * jnp.sqrt(jnp.log(time) / state.N))
