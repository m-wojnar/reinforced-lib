from functools import partial

import gymnasium as gym
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
        Sum of the rewards obtained for each arm.
    N : array_like
        Number of tries for each arm.
    """

    R: Array
    N: Array


class UCB(BaseAgent):
    r"""
    UCB agent with optional discounting. The main idea behind this algorithm is to introduce a preference factor
    in its policy, so that the selection of the next action is based on both the current estimation of the
    action-value function and the uncertainty of this estimation.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms. :math:`N \in \mathbb{N}_{+}`.
    c : float
        Degree of exploration. :math:`c \geq 0`.
    gamma : float, default=1.0
        If less than one, a discounted UCB algorithm [8]_ is used. :math:`\gamma \in (0, 1]`.

    References
    ----------
    .. [8] Aurélien Garivier, Eric Moulines. 2008. On Upper-Confidence Bound Policies for Non-Stationary
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
    def parameter_space() -> gym.spaces.Dict:
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
        Creates and initializes instance of the UCB agent for ``n_arms`` arms.  The sum of the rewards is set to zero
        and the number of tries is set to one for each arm.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of bandit arms.

        Returns
        -------
        UCBState
            Initial state of the UCB agent.
        """

        return UCBState(
            R=jnp.zeros((n_arms, 1)),
            N=jnp.ones((n_arms, 1))
        )

    @staticmethod
    def update(
        state: UCBState,
        key: PRNGKey,
        action: jnp.int32,
        reward: Scalar,
        gamma: Scalar
    ) -> UCBState:
        r"""
        In the stationary case, the sum of the rewards for a given arm is increased by reward :math:`r` obtained after
        step :math:`t` and the number of tries for the corresponding arm is incremented. In the non-stationary case,
        the update follows the equations

        .. math::
           \begin{gather}
              R_{t + 1}(a) = \mathbb{1}_{A_t = a} r + \gamma R_t(a) , \\
              N_{t + 1}(a) = \mathbb{1}_{A_t = a} + \gamma N_t(a).
           \end{gather}

        Parameters
        ----------
        state : UCBState
            Current state of agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        reward : float
            Reward collected by the agent after taking the previous action.
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
    ) -> jnp.int32:
        r"""
        UCB agent follows the policy

        .. math::
           A = \operatorname*{argmax}_{a \in \mathscr{A}} \left[ Q(a) + c \sqrt{\frac{\ln \left( {\sum_{a' \in \mathscr{A}}} N(a') \right) }{N(a)}} \right] .

        where :math:`\mathscr{A}` is a set of all actions and :math:`Q` is calculated as :math:`Q(a) = \frac{R(a)}{N(a)}`.
        The second component of the sum represents a sort of upper bound on the value of :math:`Q`, where :math:`c`
        behaves like a confidence interval and the square root - like an approximation of the :math:`Q` function
        estimation uncertainty. Note that the UCB policy is deterministic.

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
        int
            Selected action.
        """

        Q = state.R / state.N
        t = jnp.sum(state.N)

        return jnp.argmax(Q + c * jnp.sqrt(jnp.log(t) / state.N))
