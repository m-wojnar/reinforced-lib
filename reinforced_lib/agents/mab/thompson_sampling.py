from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from chex import dataclass, Array, PRNGKey, Scalar

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class ThompsonSamplingState(AgentState):
    """
    Container for the state of the Thompson sampling agent.

    Attributes
    ----------
    alpha : array_like
        Number of successful tries for each arm.
    beta : array_like
        Number of failed tries for each arm.
    """

    alpha: Array
    beta: Array


class ThompsonSampling(BaseAgent):
    r"""
    Contextual Bernoulli Thompson sampling agent with the exponential smoothing. The implementation is inspired by  the
    work of Krotov et al. [7]_. Thompson sampling is based on a beta distribution with parameters related to the number
    of successful and failed attempts. Higher values of the parameters decrease the entropy of the distribution while
    changing the ratio of the parameters shifts the expected value.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms. :math:`N \in \mathbb{N}_{+}`.
    decay : float, default=1.0
        Decay rate. If equal to zero, smoothing is not applied. :math:`w \geq 0`.

    References
    ----------
    .. [7] Alexander Krotov, Anton Kiryanov and Evgeny Khorov. 2020. Rate Control With Spatial Reuse
       for Wi-Fi 6 Dense Deployments. IEEE Access. 8. 168898-168909.
    """

    def __init__(self, n_arms: jnp.int32, decay: Scalar = 1.0) -> None:
        assert decay >= 0

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=self.n_arms))
        self.update = jax.jit(partial(self.update, decay=decay))
        self.sample = jax.jit(self.sample)

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
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
            'delta_time': gym.spaces.Box(0.0, jnp.inf, (1,))
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'context': gym.spaces.Box(-jnp.inf, jnp.inf, (self.n_arms,))
        })

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.n_arms)

    @staticmethod
    def init(key: PRNGKey, n_arms: jnp.int32) -> ThompsonSamplingState:
        r"""
        Creates and initializes an instance of the Thompson sampling agent for ``n_arms`` arms. The :math:`\mathbf{\alpha}`
        and :math:`\mathbf{\beta}` vectors are set to zero to create a non-informative prior distribution.
        The ``last_decay`` is also set to zero.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of bandit arms.

        Returns
        -------
        ThompsonSamplingState
            Initial state of the Thompson sampling agent.
        """

        return ThompsonSamplingState(
            alpha=jnp.zeros((n_arms,1)),
            beta=jnp.zeros((n_arms,1))
        )

    @staticmethod
    def update(
            state: ThompsonSamplingState,
            key: PRNGKey,
            action: jnp.int32,
            n_successful: jnp.int32,
            n_failed: jnp.int32,
            delta_time: Scalar,
            decay: Scalar
    ) -> ThompsonSamplingState:
        r"""
        Thompson sampling can be adjusted to non-stationary environments by exponential smoothing of values of
        vectors :math:`\mathbf{\alpha}` and :math:`\mathbf{\beta}` which increases the entropy of a distribution
        over time. Given a result of trial :math:`s`, we apply the following equations for each action :math:`a`:

        .. math::
           \begin{gather}
              \mathbf{\alpha}_{t + 1}(a) = \mathbf{\alpha}_t(a) e^{\frac{-\Delta t}{w}} + \mathbb{1}_{A = a} \cdot s , \\
              \mathbf{\beta}_{t + 1}(a) = \mathbf{\beta}_t(a) e^{\frac{-\Delta t}{w}} + \mathbb{1}_{A = a} \cdot (1 - s) ,
           \end{gather}

        where :math:`\Delta t` is the time elapsed since the last action selection and :math:`w` is the decay rate.

        Parameters
        ----------
        state : ThompsonSamplingState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        n_successful : int
            Number of successful tries.
        n_failed : int
            Number of failed tries.
        delta_time : float
            Time elapsed since the last action selection.
        decay : float
            Decay rate.

        Returns
        -------
        ThompsonSamplingState
            Updated agent state.
        """

        smoothing_value = jnp.exp(-decay * delta_time)

        return ThompsonSamplingState(
            alpha=(state.alpha * smoothing_value).at[action].add(n_successful),
            beta=(state.beta * smoothing_value).at[action].add(n_failed)
        )

    @staticmethod
    def sample(
            state: ThompsonSamplingState,
            key: PRNGKey,
            context: Array
    ) -> jnp.int32:
        r"""
        The Thompson sampling policy is stochastic. The algorithm draws :math:`q_a` from the distribution
        :math:`\operatorname{Beta}(1 + \mathbf{\alpha}(a), 1 + \mathbf{\beta}(a))` for each arm :math:`a`.
        The next action is selected as

        .. math::
           A = \operatorname*{argmax}_{a \in \mathscr{A}} q_a r_a ,

        where :math:`r_a` is contextual information for the arm :math:`a`, and :math:`\mathscr{A}` is a set
        of all actions.

        Parameters
        ----------
        state : ThompsonSamplingState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        context : array_like
            One-dimensional array of features for each arm.

        Returns
        -------
        int
            Selected action.
        """

        success_prob = jax.random.beta(key, 1 + state.alpha, 1 + state.beta).squeeze()
        action = jnp.argmax(success_prob * context)

        return action
