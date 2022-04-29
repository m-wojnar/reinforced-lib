from typing import NamedTuple, Callable, Tuple

import chex
import jax
import jax.numpy as jnp


class BaseAgent(NamedTuple):
    init: Callable
    update: Callable
    sample: Callable


class AgentState(NamedTuple):
    """
    alpha: number of successes for each arm
    beta: number of failures for each arm
    last_decay: time of the last decay for each arm
    """
    alpha: chex.Array
    beta: chex.Array
    last_decay: chex.Array


def thompson_sampling(context: chex.Array, decay: jnp.float32 = 1.0) -> BaseAgent:
    """
    Contextual Bernoulli Thompson sampling agent with exponential smoothing.

    :param context: one-dimensional array of arms values
    :param decay: smoothing factor (decay = 0.0 means no smoothing)
    :return: set of ThompsonSampling agent functions
    """

    def init() -> AgentState:
        """
        Creates and initializes state for ThompsonSampling agent.

        :return: initial state of ThompsonSampling agent
        """

        return AgentState(
            alpha=jnp.zeros_like(context),
            beta=jnp.zeros_like(context),
            last_decay=jnp.zeros_like(context)
        )

    def update(state: AgentState, a: jnp.int32, r: jnp.int32, time: jnp.float32 = 0.0) -> AgentState:
        """
        Updates the state of the agent after performing some action and receiving a reward.

        :param state: current state of agent
        :param a: previously selected action
        :param r: binary reward received for the previous action (1 - success, 0 - failure)
        :param time: current time
        :return: update agent state
        """

        state = _decay_one(state, a, time)
        state = AgentState(
            alpha=state.alpha.at[a].add(r),
            beta=state.beta.at[a].add(1 - r),
            last_decay=state.last_decay
        )
        return state

    def sample(state: AgentState, key: jax.random.PRNGKey, time: jnp.float32 = 0.0) -> Tuple[jnp.float32, AgentState]:
        """
        Selects next action based on current agent state.

        :param state: current state of the agent
        :param key: a PRNG key used as the random key
        :param time: current time
        :return: tuple containing selected action and updated agent state
        """

        state = _decay_all(state, time)
        success_prob = jax.random.beta(key, 1 + state.alpha, 1 + state.beta)
        action = jnp.argmax(success_prob * context)
        return action, state

    def _decay_one(state: AgentState, a: jnp.int32, time: jnp.float32) -> AgentState:
        """
        Applies exponential smoothing for values connected with action "a".

        :param state: current state of the agent
        :param a: action to apply smoothing
        :param time: current time
        :return: update agent state
        """

        smoothing_value = jnp.exp(decay * (state.last_decay[a] - time))
        state = AgentState(
            alpha=state.alpha.at[a].multiply(smoothing_value),
            beta=state.beta.at[a].multiply(smoothing_value),
            last_decay=state.last_decay.at[a].set(time)
        )
        return state

    def _decay_all(state: AgentState, time: jnp.float32) -> AgentState:
        """
        Applies exponential smoothing for all values.

        :param state: current state of the agent
        :param time: current time
        :return: update agent state
        """

        smoothing_value = jnp.exp(decay * (state.last_decay - time))
        state = AgentState(
            alpha=state.alpha * smoothing_value,
            beta=state.beta * smoothing_value,
            last_decay=jnp.full_like(context, time)
        )
        return state

    return BaseAgent(
        init,
        update,
        sample
    )
