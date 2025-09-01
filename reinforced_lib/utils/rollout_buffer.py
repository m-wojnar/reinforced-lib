from typing import Callable

import jax
import jax.numpy as jnp
from chex import dataclass, Array, Numeric, PRNGKey, Shape


@dataclass
class RolloutMemory:
    """
    Container for experience rollout buffer functions.

    Attributes
    ----------
    init : Callable
        Function that initializes the rollout buffer.
    append : Callable
        Function that appends a new values to the rollout buffer.
    compute_gae : Callable
        Function that computes advantages and returns using GAE.
    flatten_shuffle : Callable
        Function that flattens the rollout buffer and applies a random permutation.
    get_batch : Callable
        Function that gets a batch from the rollout buffer.
    """

    init: Callable
    append: Callable
    compute_gae: Callable
    flatten_shuffle: Callable
    get_batch: Callable


@dataclass
class RolloutBuffer:
    """
    Dataclass containing the rollout buffer trajectories.

    Attributes
    ----------
    states : Array
        Array containing the states.
    actions : Array
        Array containing the actions.
    rewards : Array
        Array containing the rewards.
    terminals : Array
        Array containing the terminal flags.
    values : Array
        Array containing the values computed by the value network.
    log_probs : Array
        Array containing the log probabilities of the actions.
    returns : Array
        Array containing the computed returns.
    advantages : Array
        Array containing the computed advantages.
    ptr : int
        Current pointer of the rollout buffer.
    """

    states: Array
    actions: Array
    rewards: Array
    terminals: Array
    values: Array
    log_probs: Array
    returns: Array
    advantages: Array
    ptr: int


def rollout_buffer(
        rollout_length: int,
        num_envs: int,
        batch_size: int,
        discount: float,
        lambda_gae: float,
        obs_space_shape: Shape,
        act_space_shape: Shape
) -> RolloutMemory:
    """
    Rollout buffer used for on-policy learning. Improves the stability of the learning process
    by combining the agent's experiences over multiple time steps. The buffer processes advantages
    using Generalized Advantage Estimation (GAE) _[1] and supports mini-batch updates.

    Parameters
    ----------
    rollout_length : int
        Length of the rollout buffer.
    num_envs : int
        Number of parallel environments.
    batch_size : int
        Size of the batch to be sampled from the rollout buffer.
    discount : float
        Discount factor for future rewards.
    lambda_gae : float
        GAE parameter that controls the bias-variance trade-off.
    obs_space_shape : Shape
        Shape of the observation space.
    act_space_shape : Shape
        Shape of the action space.

    Returns
    -------
    out : RolloutMemory
        Container for rollout buffer functions.

    References
    ----------
    .. [1] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous
           control using generalized advantage estimation.
    """

    def init() -> RolloutBuffer:
        """
        Initializes the rollout buffer with empty arrays.

        Returns
        -------
        buffer : RolloutBuffer
            Dataclass containing the rollout buffer trajectories.
        """

        return RolloutBuffer(
            states=jnp.empty((num_envs, rollout_length, *obs_space_shape)),
            actions=jnp.empty((num_envs, rollout_length, *act_space_shape)),
            rewards=jnp.empty((num_envs, rollout_length)),
            terminals=jnp.empty((num_envs, rollout_length), dtype=jnp.bool_),
            values=jnp.empty((num_envs, rollout_length)),
            log_probs=jnp.empty((num_envs, rollout_length)),
            returns=jnp.empty((num_envs, rollout_length)),
            advantages=jnp.empty((num_envs, rollout_length)),
            ptr=0
        )

    def append(
            buffer: RolloutBuffer,
            states: Numeric,
            actions: Numeric,
            rewards: Numeric,
            terminals: Numeric,
            values: Numeric,
            log_probs: Numeric
    ) -> RolloutBuffer:
        """
        Appends a new values to the rollout buffer.

        Parameters
        ----------
        buffer : RolloutBuffer
            Dataclass containing the rollout buffer values.
        states : Array
            State of the environment.
        actions : Array
            Action taken by the agent.
        rewards : Array
            Reward received by the agent.
        terminals : Array
            Flag indicating if the episode has terminated.
        values : Array
            Value computed by the value network.
        log_probs : Array
            Log probability of the action taken.

        Returns
        -------
        buffer : RolloutBuffer
            Updated rollout buffer.
        """

        return RolloutBuffer(
            states=buffer.states.at[:, buffer.ptr].set(states),
            actions=buffer.actions.at[:, buffer.ptr].set(actions),
            rewards=buffer.rewards.at[:, buffer.ptr].set(rewards),
            terminals=buffer.terminals.at[:, buffer.ptr].set(terminals),
            values=buffer.values.at[:, buffer.ptr].set(values),
            log_probs=buffer.log_probs.at[:, buffer.ptr].set(log_probs),
            returns=buffer.returns,
            advantages=buffer.advantages,
            ptr=buffer.ptr + 1
        )

    def compute_gae(buffer: RolloutBuffer, last_values: Array) -> RolloutBuffer:
        """
        Compute advantages and returns using GAE.

        Parameters
        ----------
        buffer : RolloutBuffer
            Rollout buffer with trajectories.
        last_values : Array
            Value predictions for the last states (one per environment).

        Returns
        -------
        buffer : RolloutBuffer
            Rollout buffer with computed advantages and returns.
        """

        values_ext = jnp.concatenate([buffer.values, last_values[:, None]], axis=1)

        def scan_fn(adv_next, step):
            rewards_t = buffer.rewards[:, step]
            terminals_t = buffer.terminals[:, step]
            values_t = values_ext[:, step]
            values_next = values_ext[:, step + 1]

            non_terminal = 1.0 - terminals_t.astype(float)
            delta = rewards_t + discount * values_next * non_terminal - values_t
            adv_t = delta + discount * lambda_gae * non_terminal * adv_next

            return adv_t, adv_t

        _, adv_seq = jax.lax.scan(scan_fn, init=jnp.zeros(num_envs), xs=jnp.arange(rollout_length), reverse=True)
        advantages = jnp.flip(adv_seq, axis=0).T
        returns = advantages + buffer.values

        return buffer.replace(advantages=advantages, returns=returns)

    def flatten_shuffle(buffer: RolloutBuffer, key: PRNGKey) -> RolloutBuffer:
        """
        Applies a random permutation to the rollout buffer.

        Parameters
        ----------
        buffer : RolloutBuffer
            Dataclass containing the rollout buffer values.
        key : PRNGKey
            Pseudo-random number generator key.

        Returns
        -------
        buffer : RolloutBuffer
            Shuffled rollout buffer.
        """

        perm = jax.random.permutation(key, num_envs * rollout_length)

        return RolloutBuffer(
            states=buffer.states.reshape((-1, *obs_space_shape))[perm],
            actions=buffer.actions.reshape((-1, *act_space_shape))[perm],
            rewards=buffer.rewards.reshape(-1)[perm],
            terminals=buffer.terminals.reshape(-1)[perm],
            values=buffer.values.reshape(-1)[perm],
            log_probs=buffer.log_probs.reshape(-1)[perm],
            returns=buffer.returns.reshape(-1)[perm],
            advantages=buffer.advantages.reshape(-1)[perm],
            ptr=0
        )

    def get_batch(buffer: RolloutBuffer, batch_idx: int) -> tuple:
        """
        Gets a batch from the rollout buffer.

        Parameters
        ----------
        buffer : RolloutBuffer
            Dataclass containing the rollout buffer values.
        batch_idx : int
            Index of the batch to be retrieved.

        Returns
        -------
        batch : tuple
            Tuple containing the states, actions, rewards, terminals, values, log probs, returns and advantages.
        """

        start_idx = batch_idx * batch_size

        states = jax.lax.dynamic_slice(buffer.states, (start_idx, *((0,) * len(obs_space_shape))), (batch_size, *obs_space_shape))
        actions = jax.lax.dynamic_slice(buffer.actions, (start_idx, *((0,) * len(act_space_shape))), (batch_size, *act_space_shape))
        rewards = jax.lax.dynamic_slice(buffer.rewards, (start_idx,), (batch_size,))
        terminals = jax.lax.dynamic_slice(buffer.terminals, (start_idx,), (batch_size,))
        values = jax.lax.dynamic_slice(buffer.values, (start_idx,), (batch_size,))
        log_probs = jax.lax.dynamic_slice(buffer.log_probs, (start_idx,), (batch_size,))
        returns = jax.lax.dynamic_slice(buffer.returns, (start_idx,), (batch_size,))
        advantages = jax.lax.dynamic_slice(buffer.advantages, (start_idx,), (batch_size,))

        return states, actions, rewards, terminals, values, log_probs, returns, advantages

    return RolloutMemory(
        init=jax.jit(init),
        append=jax.jit(append),
        compute_gae=jax.jit(compute_gae),
        flatten_shuffle=jax.jit(flatten_shuffle),
        get_batch=jax.jit(get_batch)
    )
