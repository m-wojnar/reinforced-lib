from functools import partial
from typing import Tuple

import gym.spaces
import jax
import jax.numpy as jnp
from chex import dataclass, Array, PRNGKey, Scalar

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class ThompsonSamplingState(AgentState):
    """
    Container for the state of the Thompson Sampling agent.

    Attributes
    ----------
    alpha : array_like
        Number of successful tries for each arm.
    beta : array_like
        Number of failed tries for each arm.
    last_decay : array_like
        Time of the last decay for each arm.
    """

    alpha: Array
    beta: Array
    last_decay: Array


class ThompsonSampling(BaseAgent):
    """
    Contextual Thompson Sampling agent with exponential smoothing. Implementation inspired by [1]_.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms.
    decay : float, default=1.0
        Smoothing factor (decay = 0.0 means no smoothing).

    References
    ----------
    .. [1] Krotov, Alexander & Kiryanov, Anton & Khorov, Evgeny. (2020). Rate Control With Spatial Reuse
       for Wi-Fi 6 Dense Deployments. IEEE Access. 8. 168898-168909. 10.1109/ACCESS.2020.3023552.
    """

    def __init__(self, n_arms: jnp.int32, decay: Scalar = 1.0) -> None:
        assert decay >= 0

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=self.n_arms))
        self.update = jax.jit(partial(self.update, decay=decay))
        self.sample = jax.jit(partial(self.sample, decay=decay))

    @staticmethod
    def parameters_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'decay': gym.spaces.Box(0.0, jnp.inf, (1,))
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_arms),
            'n_successful': gym.spaces.Box(0, jnp.inf, (1,), jnp.int32),
            'n_failed': gym.spaces.Box(0, jnp.inf, (1,), jnp.int32),
            'time': gym.spaces.Box(0.0, jnp.inf, (1,))
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'time': gym.spaces.Box(0.0, jnp.inf, (1,)),
            'context': gym.spaces.Box(-jnp.inf, jnp.inf, (self.n_arms,))
        })

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.n_arms)

    @staticmethod
    def init(key: PRNGKey, n_arms: jnp.int32) -> ThompsonSamplingState:
        """
        Creates and initializes instance of the Thompson Sampling agent.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of bandit arms.

        Returns
        -------
        state : ThompsonSamplingState
            Initial state of the Thompson Sampling agent.
        """

        return ThompsonSamplingState(
            alpha=jnp.zeros(n_arms),
            beta=jnp.zeros(n_arms),
            last_decay=jnp.zeros(n_arms)
        )

    @staticmethod
    def update(
            state: ThompsonSamplingState,
            key: PRNGKey,
            action: jnp.int32,
            n_successful: jnp.int32,
            n_failed: jnp.int32,
            time: Scalar,
            decay: Scalar
    ) -> ThompsonSamplingState:
        """
        Updates the state of the agent after performing some action and receiving a reward.

        Parameters
        ----------
        state : ThompsonSamplingState
            Current state of agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        n_successful : int
            Number of successful tries.
        n_failed : int
            Number of failed tries.
        time : float
            Current time.
        decay : float
            Smoothing factor.

        Returns
        -------
        state : ThompsonSamplingState
            Updated agent state.
        """

        state = ThompsonSampling._decay_one(state, action, time, decay)
        state = ThompsonSamplingState(
            alpha=state.alpha.at[action].add(n_successful),
            beta=state.beta.at[action].add(n_failed),
            last_decay=state.last_decay
        )
        return state

    @staticmethod
    def sample(
            state: ThompsonSamplingState,
            key: PRNGKey,
            time: Scalar,
            context: Array,
            decay: Scalar
    ) -> Tuple[ThompsonSamplingState, jnp.int32]:
        """
        Selects next action based on current agent state.

        Parameters
        ----------
        state : ThompsonSamplingState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        time : float
            Current time.
        context : array_like
            One-dimensional array of features for each arm.
        decay : float
            Smoothing factor.

        Returns
        -------
        tuple[ThompsonSamplingState, int]
            Tuple containing updated agent state and selected action.
        """

        state = ThompsonSampling._decay_all(state, time, decay)
        success_prob = jax.random.beta(key, 1 + state.alpha, 1 + state.beta)
        action = jnp.argmax(success_prob * context)
        return state, action

    @staticmethod
    def _decay_one(
            state: ThompsonSamplingState,
            action: jnp.int32,
            time: Scalar,
            decay: Scalar
    ) -> ThompsonSamplingState:
        """
        Applies exponential smoothing for parameters related to a given action.

        Parameters
        ----------
        state : ThompsonSamplingState
            Current state of the agent.
        action : int
            Action to apply smoothing.
        time : float
            Current time.
        decay : float
            Smoothing factor.

        Returns
        -------
        state : ThompsonSamplingState
            Updated agent state.
        """

        smoothing_value = jnp.exp(decay * (state.last_decay[action] - time))
        state = ThompsonSamplingState(
            alpha=state.alpha.at[action].multiply(smoothing_value),
            beta=state.beta.at[action].multiply(smoothing_value),
            last_decay=state.last_decay.at[action].set(time)
        )
        return state

    @staticmethod
    def _decay_all(
            state: ThompsonSamplingState,
            time: Scalar,
            decay: Scalar
    ) -> ThompsonSamplingState:
        """
        Applies exponential smoothing for parameters of all arms.

        Parameters
        ----------
        state : ThompsonSamplingState
            Current state of the agent.
        time : float
            Current time.
        decay : float
            Smoothing factor.

        Returns
        -------
        state : ThompsonSamplingState
            Updated agent state.
        """

        smoothing_value = jnp.exp(decay * (state.last_decay - time))
        state = ThompsonSamplingState(
            alpha=state.alpha * smoothing_value,
            beta=state.beta * smoothing_value,
            last_decay=jnp.full_like(state.last_decay, time)
        )
        return state
