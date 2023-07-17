from typing import Callable

import jax
import jax.numpy as jnp
from chex import dataclass, Array, Numeric, Scalar, PRNGKey, Shape


@dataclass
class ExperienceReplay:
    """
    Container for experience replay buffer functions.

    Attributes
    ----------
    init : Callable
        Function that initializes the replay buffer.
    append : Callable
        Function that appends a new values to the replay buffer.
    sample : Callable
        Function that samples a batch from the replay buffer.
    is_ready : Callable
        Function that checks if the replay buffer is ready to be sampled.
    """

    init: Callable
    append: Callable
    sample: Callable
    is_ready: Callable


@dataclass
class ReplayBuffer:
    """
    Dataclass containing the replay buffer values. The replay buffer is implemented as a circular buffer.

    Attributes
    ----------
    states : array_like
        Array containing the states.
    actions : array_like
        Array containing the actions.
    rewards : array_like
        Array containing the rewards.
    terminals : array_like
        Array containing the terminal flags.
    next_states : array_like
        Array containing the next states.
    size : int
        Current size of the replay buffer.
    ptr : int
        Current pointer of the replay buffer.
    """

    states: Array
    actions: Array
    rewards: Array
    terminals: Array
    next_states: Array
    size: jnp.int32
    ptr: jnp.int32


def experience_replay(
        buffer_size: jnp.int32,
        batch_size: jnp.int32,
        obs_space_shape: Shape,
        act_space_shape: Shape
) -> ExperienceReplay:
    """
    Experience replay buffer used for off-policy learning. Improves the stability of the learning process
    by reducing the correlation between the samples and enables an agent to learn from past experiences.

    Parameters
    ----------
    buffer_size : int
        Maximum size of the replay buffer.
    batch_size : int
        Size of the batch to be sampled from the replay buffer.
    obs_space_shape : Shape
        Shape of the observation space.
    act_space_shape : Shape
        Shape of the action space.

    Returns
    -------
    out : ExperienceReplay
        Container for experience replay buffer functions.
    """

    def init() -> ReplayBuffer:
        """
        Initializes the replay buffer with empty arrays.

        Returns
        -------
        buffer : ReplayBuffer
            Dataclass containing the replay buffer values.
        """

        return ReplayBuffer(
            states=jnp.empty((buffer_size, *obs_space_shape)),
            actions=jnp.empty((buffer_size, *act_space_shape)),
            rewards=jnp.empty((buffer_size, 1)),
            terminals=jnp.empty((buffer_size, 1), dtype=jnp.bool_),
            next_states=jnp.empty((buffer_size, *obs_space_shape)),
            size=0,
            ptr=0
        )

    def append(
            buffer: ReplayBuffer,
            state: Numeric,
            action: Numeric,
            reward: Scalar,
            terminal: jnp.bool_,
            next_state: Numeric
    ) -> ReplayBuffer:
        """
        Appends a new values to the replay buffer.

        Parameters
        ----------
        buffer : ReplayBuffer
            Dataclass containing the replay buffer values.
        state : array_like
            State of the environment.
        action : array_like
            Action taken by the agent.
        reward : float
            Reward received by the agent.
        terminal : bool
            Flag indicating if the episode has terminated.
        next_state : array_like
            Next state of the environment.

        Returns
        -------
        buffer : ReplayBuffer
            Updated replay buffer.
        """

        return ReplayBuffer(
            states=buffer.states.at[buffer.ptr].set(state),
            actions=buffer.actions.at[buffer.ptr].set(action),
            rewards=buffer.rewards.at[buffer.ptr].set(reward),
            terminals=buffer.terminals.at[buffer.ptr].set(terminal),
            next_states=buffer.next_states.at[buffer.ptr].set(next_state),
            size=jax.lax.min(buffer.size + 1, buffer_size),
            ptr=(buffer.ptr + 1) % buffer_size
        )

    def sample(buffer: ReplayBuffer, key: PRNGKey) -> tuple:
        """
        Samples a batch from the replay buffer (there may be duplicates!).

        Parameters
        ----------
        buffer : ReplayBuffer
            Dataclass containing the replay buffer values.
        key : PRNGKey
            Pseudo-random number generator key.

        Returns
        -------
        batch : tuple
            Tuple containing the batch of states, actions, rewards, terminals and next states.
        """

        idxs = jax.random.uniform(key, shape=(batch_size,), minval=0, maxval=buffer.size).astype(jnp.int32)

        states = buffer.states[idxs]
        actions = buffer.actions[idxs]
        rewards = buffer.rewards[idxs]
        terminals = buffer.terminals[idxs]
        next_states = buffer.next_states[idxs]

        return states, actions, rewards, terminals, next_states

    def is_ready(buffer: ReplayBuffer) -> jnp.bool_:
        """
        Checks if the replay buffer is ready to be sampled (contains at least ``batch_size`` elements).

        Parameters
        ----------
        buffer : ReplayBuffer
            Dataclass containing the replay buffer values.

        Returns
        -------
        ready : bool
            Flag indicating if the replay buffer is ready to be sampled.
        """

        return buffer.size >= batch_size

    return ExperienceReplay(
        init=jax.jit(init),
        append=jax.jit(append),
        sample=jax.jit(sample),
        is_ready=jax.jit(is_ready)
    )
