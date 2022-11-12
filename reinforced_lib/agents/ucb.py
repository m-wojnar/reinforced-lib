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
    UCB agent with optional discounting.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms.
    c : float
        Degree of exploration.
    gamma : float, default=0.0
        If non-zero than discounted UCB algorithm [5]_ is used. ``gamma`` must be in [0, 1).

    References
    ----------
    .. [5] Garivier, A., & Moulines, E. (2008). On Upper-Confidence Bound Policies for Non-Stationary
       Bandit Problems. 10.48550/ARXIV.0805.3415.
    """

    def __init__(
            self,
            n_arms: jnp.int32,
            c: Scalar,
            gamma: Scalar = 0.0
    ) -> None:
        assert c >= 0
        assert 0 <= gamma < 1

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=n_arms))
        self.update = jax.jit(partial(self.update, gamma=gamma))
        self.sample = jax.jit(partial(self.sample, c=c))

    @staticmethod
    def parameters_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'c': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32),
            'gamma': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32)
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_arms),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), jnp.float32)
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
            N=jnp.ones(n_arms)
        )

    @staticmethod
    def update(
        state: UCBState,
        key: PRNGKey,
        action: jnp.int32,
        reward: Scalar,
        gamma: Scalar
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
        gamma : float
            Discount factor (used when ``gamma > 0``).

        Returns
        -------
        UCBState
            Updated agent state.
        """

        def classic_update(operands: Tuple) -> UCBState:
            state, action, reward, _ = operands
            return UCBState(
                Q=state.Q.at[action].add((reward - state.Q[action]) / state.N[action]),
                N=state.N.at[action].add(1)
            )

        def discounted_update(operands: Tuple) -> UCBState:
            state, action, reward, gamma = operands
            N_prev = state.N
            N = (gamma * state.N).at[action].add(1)
            return UCBState(
                Q=(jnp.zeros_like(state.Q).at[action].set(reward) + gamma * N_prev * state.Q) / N,
                N=N
            )

        return jax.lax.cond(gamma == 0, classic_update, discounted_update, (state, action, reward, gamma))

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
