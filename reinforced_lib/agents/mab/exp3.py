from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from chex import dataclass, Array, Scalar, PRNGKey

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class Exp3State(AgentState):
    """
    Container for the state of the Exp3 agent.

    Attributes
    ----------
    omega : array_like
        Preference for each arm.
    """

    omega: Array


class Exp3(BaseAgent):
    r"""
    Basic Exp3 agent for stationary multi-armed bandit problems with exploration factor :math:`\gamma`. The higher
    the value, the more the agent explores. The implementation is inspired by the work of Auer et al. [6]_. There
    are many variants of the Exp3 algorithm, you can find more information in the original paper.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms. :math:`N \in \mathbb{N}_{+}`.
    gamma : float
        Exploration factor. :math:`\gamma \in (0, 1]`.
    min_reward : float
        Minimum possible reward.
    max_reward : float
        Maximum possible reward.

    References
    ----------
    .. [6] Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. 2002. The Nonstochastic Multiarmed
       Bandit Problem. SIAM Journal on Computing, 32(1), 48–77.
    """

    def __init__(
            self,
            n_arms: jnp.int32,
            gamma: Scalar,
            min_reward: Scalar,
            max_reward: Scalar
    ) -> None:
        assert 0 < gamma <= 1

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=n_arms))
        self.update = jax.jit(partial(self.update, gamma=gamma, min_reward=min_reward, max_reward=max_reward))
        self.sample = jax.jit(partial(self.sample, gamma=gamma))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'gamma': gym.spaces.Box(0, 1, (1,), jnp.float32),
            'min_reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), jnp.float32),
            'max_reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), jnp.float32)
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
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.n_arms)

    @staticmethod
    def init(
            key: PRNGKey,
            n_arms: jnp.int32
    ) -> Exp3State:
        """
        Initializes the Exp3 agent state with uniform preference for each arm.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of bandit arms.

        Returns
        -------
        Exp3State
            Initial state of the Exp3 agent.
        """

        return Exp3State(
            omega=jnp.ones((n_arms, 1)) / n_arms
        )

    @staticmethod
    def update(
        state: Exp3State,
        key: PRNGKey,
        action: jnp.int32,
        reward: Scalar,
        gamma: Scalar,
        min_reward: Scalar,
        max_reward: Scalar
    ) -> Exp3State:
        r"""
        Agent updates its preference for the selected arm :math:`a` according to the following formula:

        .. math::
            \omega_{t + 1}(a) = \omega_{t}(a) \exp \left( \frac{\gamma r}{\pi(a) K} \right)

        where :math:`\omega_{t + 1}(a)` is the preference of arm :math:`a` at time :math:`t + 1`, :math:`\pi(a)` is
        the probability of selecting arm :math:`a`, and :math:`K` is the number of arms. The reward :math:`r` is
        normalized to the range :math:`[0, 1]`. The exponential growth significantly increases the weight of good arms,
        so in the long use of the agent it is important to **ensure that the values of** :math:`\omega` **do not exceed
        the maximum value of the floating point type!**

        Parameters
        ----------
        state : Exp3State
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        reward : float
            Reward collected by the agent after taking the previous action.
        gamma : float
            Exploration factor.
        min_reward : float
            Minimum possible reward.
        max_reward : float
            Maximum possible reward.

        Returns
        -------
        Exp3State
            Updated agent state.
        """

        reward = (reward - min_reward) / (max_reward - min_reward)
        n_arms = state.omega.size

        pi = (1 - gamma) * state.omega / state.omega.sum() + gamma / n_arms

        return Exp3State(
            omega=state.omega.at[action].mul(jnp.exp(gamma * reward / (pi[action] * n_arms)))
        )

    @staticmethod
    def sample(
        state: Exp3State,
        key: PRNGKey,
        gamma: Scalar
    ) -> jnp.int32:
        r"""
        The Exp3 policy is stochastic. Algorithm chooses a random arm with probability :math:`\gamma`, otherwise it
        draws arm :math:`a` with probability :math:`\omega(a) / \sum_{b=1}^N \omega(b)`.

        Parameters
        ----------
        state : Exp3State
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        gamma : float
            Exploration factor.

        Returns
        -------
        int
            Selected action.
        """

        pi = (1 - gamma) * state.omega / state.omega.sum() + gamma / state.omega.size
        return jax.random.categorical(key, jnp.log(pi.squeeze()))
