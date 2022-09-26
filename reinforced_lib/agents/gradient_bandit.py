from functools import partial
from typing import Tuple

import gym.spaces
import jax
import jax.numpy as jnp
from chex import dataclass, Array, Scalar, PRNGKey

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class GradientBanditState(AgentState):
    """
    Container for the state of the gradient bandit agent.

    Attributes
    ----------
    H : array_like
        Preference for each arm.
    R : float
        Average of rewards for each arm.
    n : int
        Number of the step.
    """

    H: Array
    R: Scalar
    n: jnp.int64


class GradientBandit(BaseAgent):
    """
    Gradient bandit agent with baseline and optional exponential recency-weighted average update.
    Implementation inspired by [4]_.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms.
    lr : float
        Step size.
    alpha : float, default=0.0
        If non-zero than exponential recency-weighted average is used to update Q values. ``alpha`` must be in [0, 1).

    References
    ----------
    .. [4]  Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction. The MIT Press. 37-40.
    """

    def __init__(
            self,
            n_arms: jnp.int32,
            lr: Scalar,
            alpha: Scalar = 0.0
    ) -> None:
        assert lr > 0
        assert 0 <= alpha <= 1

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=n_arms))
        self.update = jax.jit(partial(self.update, lr=lr, alpha=alpha))
        self.sample = jax.jit(self.sample)

    @staticmethod
    def parameters_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'lr': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32),
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
            n_arms: jnp.int32
    ) -> GradientBanditState:
        """
        Creates and initializes instance of the gradient bandit agent.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of contextual bandit arms.

        Returns
        -------
        state : GradientBanditState
            Initial state of the gradient bandit agent.
        """

        return GradientBanditState(
            H=jnp.zeros(n_arms),
            R=0.0,
            n=1
        )

    @staticmethod
    def update(
        state: GradientBanditState,
        key: PRNGKey,
        action: jnp.int32,
        reward: Scalar,
        lr: Scalar,
        alpha: Scalar
    ) -> GradientBanditState:
        """
        Updates the state of the agent after performing some action and receiving a reward.

        Parameters
        ----------
        state : GradientBanditState
            Current state of agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        reward : float
            Reward as a result of previous action.
        lr : float
            Step size.
        alpha : float
            Exponential recency-weighted average factor (used when ``alpha > 0``).

        Returns
        -------
        GradientBanditState
            Updated agent state.
        """

        R = jnp.where(state.n == 1, reward, state.R)
        mask = jnp.ones_like(state.H, dtype=jnp.bool_).at[action].set(False)
        pi = jax.nn.softmax(state.H)

        return GradientBanditState(
            H=state.H + lr * (reward - R) * jnp.where(mask, -pi, 1 - pi),
            R=R + (reward - R) * jnp.where(alpha == 0, 1 / state.n, alpha),
            n=state.n + 1
        )

    @staticmethod
    def sample(
        state: GradientBanditState,
        key: PRNGKey
    ) -> Tuple[GradientBanditState, jnp.int32]:
        """
        Selects next action based on current agent state.

        Parameters
        ----------
        state : GradientBanditState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.

        Returns
        -------
        tuple[GradientBanditState, jnp.int32]
            Tuple containing updated agent state and selected action.
        """

        return state, jax.random.categorical(key, state.H)
