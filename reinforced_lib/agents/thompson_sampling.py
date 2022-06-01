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
        Number of successes for each arm.
    beta : chex.Array
        Number of failures for each arm.
    last_decay : chex.Array
        Time of the last decay for each arm.
    """

    alpha: chex.Array
    beta: chex.Array
    last_decay: chex.Array


def thompson_sampling(context: chex.Array, decay: chex.Scalar = 1.0) -> BaseAgent:
    """
    Contextual Bernoulli Thompson Sampling agent with exponential smoothing.

    Parameters
    ----------
    context : chex.Array
        One-dimensional array of arms values.
    decay : float
        Smoothing factor (decay = 0.0 means no smoothing).

    Returns
    -------
    out : BaseAgent
        Container for functions of the Thompson Sampling agent.
    """

    def update_observation_space() -> gym.spaces.Dict:
        """
        Defines parameters required by the agents 'update' function.

        Returns
        -------
        out: gym.spaces.Dict
            Parameters in OpenAI Gym format.
        """

        return gym.spaces.Dict({
            'a': gym.spaces.Discrete(len(context)),
            'r': gym.spaces.Discrete(2),
            'time': gym.spaces.Box(0.0, float('inf'), shape=(1,))
        })

    def sample_observation_space() -> gym.spaces.Dict:
        """
        Defines parameters required by the agents 'sample' function.

        Returns
        -------
        out: gym.spaces.Dict
            Parameters in OpenAI Gym format.
        """

        return gym.spaces.Dict({
            'key': gym.spaces.Space(dtype=chex.PRNGKey),
            'time': gym.spaces.Box(0.0, float('inf'), shape=(1,))
        })

    def action_space() -> gym.spaces.Tuple:
        """
        Defines action returned by the agent.

        Returns
        -------
        out: gym.spaces.Space
            Action in OpenAI Gym format.
        """

        return gym.spaces.Tuple((gym.spaces.Discrete(len(context)),))

    def init() -> ThompsonSamplingState:
        """
        Creates and initializes state for the Thompson Sampling agent.

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

    def update(state: ThompsonSamplingState, a: jnp.int32, r: jnp.int32,
               time: chex.Scalar = 0.0) -> ThompsonSamplingState:
        """
        Updates the state of the agent after performing some action and receiving a reward.

        Parameters
        ----------
        state : ThompsonSamplingState
            Current state of agent.
        a : int
            Previously selected action.
        r : int or bool
            Binary reward received for the previous action (1 - success, 0 - failure).
        time : float
            Current time.

        Returns
        -------
        out : ThompsonSamplingState
            Updated agent state.
        """

        state = _decay_one(state, a, time)
        state = ThompsonSamplingState(
            alpha=state.alpha.at[a].add(r),
            beta=state.beta.at[a].add(1 - r),
            last_decay=state.last_decay
        )
        return state

    def sample(state: ThompsonSamplingState, key: chex.PRNGKey,
               time: chex.Scalar = 0.0) -> Tuple[ThompsonSamplingState, jnp.int32]:
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

        Returns
        -------
        out : Tuple[ThompsonSamplingState, int]
            Tuple containing updated agent state and selected action.
        """

        state = _decay_all(state, time)
        success_prob = jax.random.beta(key, 1 + state.alpha, 1 + state.beta)
        action = jnp.argmax(success_prob * context)
        return state, action

    def _decay_one(state: ThompsonSamplingState, a: jnp.int32, time: chex.Scalar) -> ThompsonSamplingState:
        """
        Applies exponential smoothing for parameters related to action 'a'.

        Parameters
        ----------
        state : ThompsonSamplingState
            Current state of the agent.
        a : int
            Action to apply smoothing.
        time : float
            Current time.

        Returns
        -------
        out : ThompsonSamplingState
            Updated agent state.
        """

        smoothing_value = jnp.exp(decay * (state.last_decay[a] - time))
        state = ThompsonSamplingState(
            alpha=state.alpha.at[a].multiply(smoothing_value),
            beta=state.beta.at[a].multiply(smoothing_value),
            last_decay=state.last_decay.at[a].set(time)
        )
        return state

    def _decay_all(state: ThompsonSamplingState, time: chex.Scalar) -> ThompsonSamplingState:
        """
        Applies exponential smoothing for all parameters.

        Parameters
        ----------
        state : ThompsonSamplingState
            Current state of the agent.
        time : float
            Current time.

        Returns
        -------
        out : ThompsonSamplingState
            Updated agent state.
        """

        smoothing_value = jnp.exp(decay * (state.last_decay - time))
        state = ThompsonSamplingState(
            alpha=state.alpha * smoothing_value,
            beta=state.beta * smoothing_value,
            last_decay=jnp.full_like(context, time)
        )
        return state

    return BaseAgent(
        init,
        update,
        sample,
        update_observation_space,
        sample_observation_space,
        action_space
    )
