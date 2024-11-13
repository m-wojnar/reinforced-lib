from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from chex import dataclass, Array, PRNGKey, Scalar

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class NormalThompsonSamplingState(AgentState):
    """
    Container for the state of the normal Thompson sampling agent.

    Attributes
    ----------
    alpha : Array
        The concentration parameter of the inverse-gamma distribution.
    beta : Array
        The scale parameter of the inverse-gamma distribution.
    lam : Array
        The number of observations.
    mu : Array
        The mean of the normal distribution.
    """

    alpha: Array
    beta: Array
    lam: Array
    mu: Array


class NormalThompsonSampling(BaseAgent):
    r"""
    Normal Thompson sampling agent [10]_. The normal-inverse-gamma distribution is a conjugate prior for the normal
    distribution with unknown mean and variance. The parameters of the distribution are updated after each observation.
    The mean of the normal distribution is sampled from the normal-inverse-gamma distribution and the action with
    the highest expected value is selected.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms. :math:`N \in \mathbb{N}_{+}` .
    alpha : float
        See also ``NormalThompsonSamplingState`` for interpretation. :math:`\alpha > 0`.
    beta : float
        See also ``NormalThompsonSamplingState`` for interpretation. :math:`\beta > 0`.
    lam : float
        See also ``NormalThompsonSamplingState`` for interpretation. :math:`\lambda > 0`.
    mu : float
        See also ``NormalThompsonSamplingState`` for interpretation. :math:`\mu \in \mathbb{R}`.

    References
    ----------
    .. [10] Kevin P. Murphy. 2007. Conjugate Bayesian analysis of the Gaussian distribution.
       https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    """

    def __init__(
            self,
            n_arms: int,
            alpha: Scalar,
            beta: Scalar,
            lam: Scalar,
            mu: Scalar
    ) -> None:
        assert alpha > 0
        assert beta > 0
        assert lam >= 0

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=self.n_arms, alpha=alpha, beta=beta, lam=lam, mu=mu))
        self.update = jax.jit(self.update)
        self.sample = jax.jit(self.sample)

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), int),
            'alpha': gym.spaces.Box(0.0, jnp.inf, (1,), float),
            'beta': gym.spaces.Box(0.0, jnp.inf, (1,), float),
            'lam': gym.spaces.Box(0.0, jnp.inf, (1,), float),
            'mu': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_arms),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)
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
            n_arms: int,
            alpha: Scalar,
            beta: Scalar,
            lam: Scalar,
            mu: Scalar
    ) -> NormalThompsonSamplingState:
        r"""
        Creates and initializes an instance of the normal Thompson sampling agent for ``n_arms`` arms and the
        given initial parameters for the prior distribution.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of bandit arms.
        alpha : float
            See also ``NormalThompsonSamplingState`` for interpretation.
        beta : float
            See also ``NormalThompsonSamplingState`` for interpretation.
        lam : float
            See also ``NormalThompsonSamplingState`` for interpretation.
        mu : float
            See also ``NormalThompsonSamplingState`` for interpretation.
        Returns
        -------
        NormalThompsonSamplingState
            Initial state of the normal Thompson sampling agent.
        """

        return NormalThompsonSamplingState(
            alpha=jnp.full((n_arms, 1), alpha),
            beta=jnp.full((n_arms, 1), beta),
            lam=jnp.full((n_arms, 1), lam),
            mu=jnp.full((n_arms, 1), mu)
        )

    @staticmethod
    def update(
            state: NormalThompsonSamplingState,
            key: PRNGKey,
            action: int,
            reward: Scalar
    ) -> NormalThompsonSamplingState:
        r"""
        Normal Thompson sampling update according to [10]_.

        .. math::
          \begin{align}
            \alpha_{t + 1}(a) &= \alpha_t(a) + \frac{1}{2} \\
            \beta_{t + 1}(a) &= \beta_t(a) + \frac{\lambda_t(a) (r_t(a) - \mu_t(a))^2}{2 (\lambda_t(a) + 1)} \\
            \lambda_{t + 1}(a) &= \lambda_t(a) + 1 \\
            \mu_{t + 1}(a) &= \frac{\mu_t(a) \lambda_t(a) + r_t(a)}{\lambda_t(a) + 1}
          \end{align}

        Parameters
        ----------
        state : NormalThompsonSamplingState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        reward : Float
            Reward obtained upon execution of action.

        Returns
        -------
        NormalThompsonSamplingState
            Updated agent state.
        """

        lam = state.lam[action]
        mu = state.mu[action]

        return NormalThompsonSamplingState(
            alpha=state.alpha.at[action].add(1 / 2),
            beta=state.beta.at[action].add((lam * jnp.square(reward - mu)) / (2 * (lam + 1))),
            lam=state.lam.at[action].add(1),
            mu=state.mu.at[action].set((mu * lam + reward) / (lam + 1))
        )

    @staticmethod
    def inverse_gamma(key: PRNGKey, concentration: Array, scale: Array) -> Array:
        r"""
        Samples from the inverse gamma distribution. Implementation is based on the gamma distribution and the
        following dependence:

        .. math::
          \begin{gather}
              X \sim \operatorname{Gamma}(\alpha, \beta) \\
              \frac{1}{X} \sim \operatorname{Inverse-gamma}(\alpha, \frac{1}{\beta})
          \end{gather}

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        concentration : Array
            The concentration parameter of the inverse-gamma distribution.
        scale : Array
            The scale parameter of the inverse-gamma distribution.

        Returns
        -------
        Array
            Sampled values from the inverse gamma distribution.
        """

        gamma = jax.random.gamma(key, concentration) / scale
        return 1 / gamma

    @staticmethod
    def sample(state: NormalThompsonSamplingState, key: PRNGKey) -> int:
        r"""
        The normal Thompson sampling policy is stochastic. The algorithm draws :math:`q_a` from the distribution
        :math:`\operatorname{Normal}(\mu(a), \operatorname{scale}(a)/\sqrt{\lambda(a)})` for each arm :math:`a` where
        :math:`\text{scale}(a)` is sampled from the inverse gamma distribution with parameters :math:`\alpha(a)` and
        :math:`\beta(a)`. The next action is selected as :math:`A = \operatorname*{argmax}_{a \in \mathscr{A}} q_a`,
        where :math:`\mathscr{A}` is a set of all actions.

        Parameters
        ----------
        state : NormalThompsonSamplingState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.

        Returns
        -------
        int
            Selected action.
        """

        loc_key, scale_key = jax.random.split(key)

        scale = jnp.sqrt(NormalThompsonSampling.inverse_gamma(scale_key, state.alpha, state.beta))
        loc = state.mu + jax.random.normal(loc_key, shape=state.mu.shape) * scale / jnp.sqrt(state.lam)

        return jnp.argmax(loc)
