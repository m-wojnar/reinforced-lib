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
    R : array_like
        Sum of rewards obtained for each arm
    N : array_like
        Number of tries for each arm
    """

    R: Array
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
    gamma : float, default=1.0
        If less than one, discounted UCB algorithm [5]_ is used. ``gamma`` must be in (0, 1].

    References
    ----------
    .. [5] Garivier, A., & Moulines, E. (2008). On Upper-Confidence Bound Policies for Non-Stationary
       Bandit Problems. 10.48550/ARXIV.0805.3415.
    """

    def __init__(
            self,
            n_arms: jnp.int32,
            c: Scalar,
            gamma: Scalar = 1.0
    ) -> None:
        assert c >= 0
        assert 0 < gamma <= 1

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
        return gym.spaces.Dict({})

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
            R=jnp.zeros(n_arms),
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
            Discount factor.

        Returns
        -------
        UCBState
            Updated agent state.
        """

        return UCBState(
            R=(gamma * state.R).at[action].add(reward),
            N=(gamma * state.N).at[action].add(1)
        )

    @staticmethod
    def sample(
        state: UCBState,
        key: PRNGKey,
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
        c : float
            Degree of exploration.

        Returns
        -------
        tuple[UCBState, jnp.int32]
            Tuple containing updated agent state and selected action.
        """

        Q = state.R / state.N
        t = jnp.sum(state.N)
        return state, jnp.argmax(Q + c * jnp.sqrt(jnp.log(t) / state.N))
