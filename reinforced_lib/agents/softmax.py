from functools import partial
from typing import Tuple

import gym.spaces
import jax
import jax.numpy as jnp
from chex import dataclass, Array, Scalar, PRNGKey

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class SoftmaxState(AgentState):
    """
    Container for the state of the Softmax agent.

    Attributes
    ----------
    H : array_like
        Preference for each arm.
    r : float
        Average of all obtained rewards.
    n : int
        Number of the step.
    """

    H: Array
    r: Scalar
    n: jnp.int64


class Softmax(BaseAgent):
    """
    Softmax agent with baseline and optional exponential recency-weighted average update.
    Algorithms policy can be controlled by temperature parameter. Implementation inspired by [4]_.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms.
    lr : float
        Step size. ``lr`` must be greater than 0.
    alpha : float, default=0.0
        If non-zero than exponential recency-weighted average is used to update Q values. ``alpha`` must be in [0, 1].
    tau : float, default=1.0
        Temperature parameter. ``tau`` must be greater than 0.

    References
    ----------
    .. [4]  Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction. The MIT Press. 37-40.
    """

    def __init__(
            self,
            n_arms: jnp.int32,
            lr: Scalar,
            alpha: Scalar = 0.0,
            tau: Scalar = 1.0
    ) -> None:
        assert lr > 0
        assert 0 <= alpha <= 1
        assert tau > 0

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=n_arms))
        self.update = jax.jit(partial(self.update, lr=lr, alpha=alpha))
        self.sample = jax.jit(partial(self.sample, tau=tau))

    @staticmethod
    def parameters_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'lr': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32),
            'alpha': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32),
            'tau': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32)
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
    ) -> SoftmaxState:
        """
        Creates and initializes instance of the Softmax agent.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of contextual bandit arms.

        Returns
        -------
        state : SoftmaxState
            Initial state of the Softmax agent.
        """

        return SoftmaxState(
            H=jnp.zeros(n_arms),
            r=0.0,
            n=1
        )

    @staticmethod
    def update(
        state: SoftmaxState,
        key: PRNGKey,
        action: jnp.int32,
        reward: Scalar,
        lr: Scalar,
        alpha: Scalar
    ) -> SoftmaxState:
        """
        Updates the state of the agent after performing some action and receiving a reward.

        Parameters
        ----------
        state : SoftmaxState
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
        SoftmaxState
            Updated agent state.
        """

        r = jnp.where(state.n == 1, reward, state.r)
        mask = jnp.ones_like(state.H, dtype=jnp.bool_).at[action].set(False)
        pi = jax.nn.softmax(state.H)

        return SoftmaxState(
            H=state.H + lr * (reward - r) * jnp.where(mask, -pi, 1 - pi),
            r=r + (reward - r) * jnp.where(alpha == 0, 1 / state.n, alpha),
            n=state.n + 1
        )

    @staticmethod
    def sample(
        state: SoftmaxState,
        key: PRNGKey,
        tau: Scalar
    ) -> Tuple[SoftmaxState, jnp.int32]:
        """
        Selects next action based on current agent state.

        Parameters
        ----------
        state : SoftmaxState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        tau : float
            Temperature parameter.

        Returns
        -------
        tuple[SoftmaxState, jnp.int32]
            Tuple containing updated agent state and selected action.
        """

        return state, jax.random.categorical(key, state.H / tau)
