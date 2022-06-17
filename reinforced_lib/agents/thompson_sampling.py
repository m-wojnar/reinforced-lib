from functools import partial
from typing import Tuple

import chex
import gym.spaces
import jax
import jax.numpy as jnp

from reinforced_lib.agents.agent_state import AgentState
from reinforced_lib.agents.base_agent import BaseAgent


@chex.dataclass
class ThompsonSamplingState(AgentState):
    """
    Container for the state of the Thompson Sampling agent.

    Fields
    ------
    alpha : chex.Array
        Number of successful tries for each arm.
    beta : chex.Array
        Number of failed tries for each arm.
    last_decay : chex.Array
        Time of the last decay for each arm.
    """

    alpha: chex.Array
    beta: chex.Array
    last_decay: chex.Array


def init(context: chex.Array) -> ThompsonSamplingState:
    """
    Creates and initializes instance of the Thompson Sampling agent.

    Parameters
    ----------
    context : chex.Array
        One-dimensional array of features of each arm.

    Returns
    -------
    out : ThompsonSamplingState
        Initial state of the Thompson Sampling agent.
    """

    return ThompsonSamplingState(
        alpha=jnp.zeros_like(context),
        beta=jnp.zeros_like(context),
        last_decay=jnp.zeros_like(context)
    )


def update(
        state: ThompsonSamplingState,
        action: jnp.int32,
        n_successful: jnp.int32,
        n_failed: jnp.int32,
        time: chex.Scalar,
        decay: chex.Scalar
) -> ThompsonSamplingState:
    """
    Updates the state of the agent after performing some action and receiving a reward.

    Parameters
    ----------
    state : ThompsonSamplingState
        Current state of agent.
    action : int
        Previously selected action.
    n_successful : int
        Number of successful tries.
    n_failed : int
        Number of failed tries.
    time : float
        Current time.
    decay : float
        Smoothing factor (decay = 0.0 means no smoothing).

    Returns
    -------
    out : ThompsonSamplingState
        Updated agent state.
    """

    state = decay_one(state, action, time, decay)
    state = ThompsonSamplingState(
        alpha=state.alpha.at[action].add(n_successful),
        beta=state.beta.at[action].add(n_failed),
        last_decay=state.last_decay
    )
    return state


def sample(
        state: ThompsonSamplingState,
        key: chex.PRNGKey,
        time: chex.Scalar,
        context: chex.Array,
        decay: chex.Scalar
) -> Tuple[ThompsonSamplingState, jnp.int32]:
    """
    Selects next action based on current agent state.

    Parameters
    ----------
    state : ThompsonSamplingState
        Current state of the agent.
    key : chex.PRNGKey
        A PRNG key used as the random key.
    time : float
        Current time.
    context : chex.Array
        One-dimensional array of features of each arm.
    decay : float
        Smoothing factor (decay = 0.0 means no smoothing).

    Returns
    -------
    out : Tuple[ThompsonSamplingState, int]
        Tuple containing updated agent state and selected action.
    """

    state = decay_all(state, time, decay)
    success_prob = jax.random.beta(key, 1 + state.alpha, 1 + state.beta)
    action = jnp.argmax(success_prob * context)
    return state, action


def decay_one(
        state: ThompsonSamplingState,
        action: jnp.int32,
        time: chex.Scalar,
        decay: chex.Scalar
) -> ThompsonSamplingState:
    """
    Applies exponential smoothing for parameters related to action 'a'.

    Parameters
    ----------
    state : ThompsonSamplingState
        Current state of the agent.
    action : int
        Action to apply smoothing.
    time : float
        Current time.
    decay : float
        Smoothing factor (decay = 0.0 means no smoothing).

    Returns
    -------
    out : ThompsonSamplingState
        Updated agent state.
    """

    smoothing_value = jnp.exp(decay * (state.last_decay[action] - time))
    state = ThompsonSamplingState(
        alpha=state.alpha.at[action].multiply(smoothing_value),
        beta=state.beta.at[action].multiply(smoothing_value),
        last_decay=state.last_decay.at[action].set(time)
    )
    return state


def decay_all(
        state: ThompsonSamplingState,
        time: chex.Scalar,
        decay: chex.Scalar
) -> ThompsonSamplingState:
    """
    Applies exponential smoothing for all parameters.

    Parameters
    ----------
    state : ThompsonSamplingState
        Current state of the agent.
    time : float
        Current time.
    decay : float
        Smoothing factor (decay = 0.0 means no smoothing).

    Returns
    -------
    out : ThompsonSamplingState
        Updated agent state.
    """

    smoothing_value = jnp.exp(decay * (state.last_decay - time))
    state = ThompsonSamplingState(
        alpha=state.alpha * smoothing_value,
        beta=state.beta * smoothing_value,
        last_decay=jnp.full_like(state.last_decay, time)
    )
    return state


class ThompsonSampling(BaseAgent):
    """
    Contextual Thompson Sampling agent with exponential smoothing.

    Parameters
    ----------
    context : chex.Array
        One-dimensional array of arms values.
    decay : float
        Smoothing factor (decay = 0.0 means no smoothing).
    """

    def __init__(self, context: chex.Array, decay: chex.Scalar = 1.0):
        self.context_len = len(context)

        self.init = jax.jit(partial(init, context=context))
        self.update = jax.jit(partial(update, decay=decay))
        self.sample = jax.jit(partial(sample, context=context, decay=decay))

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.context_len),
            'n_successful': gym.spaces.Box(0, jnp.inf, (1,), jnp.int32),
            'n_failed': gym.spaces.Box(0, jnp.inf, (1,), jnp.int32),
            'time': gym.spaces.Box(0.0, jnp.inf, (1,))
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'time': gym.spaces.Box(0.0, jnp.inf, (1,))
        })

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.context_len)
