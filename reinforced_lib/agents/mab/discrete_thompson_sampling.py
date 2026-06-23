from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from chex import dataclass, Array, PRNGKey, Scalar

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class DiscreteThompsonSamplingState(AgentState):
    r"""
    Container for the state of the discrete Thompson sampling agent.

    Attributes
    ----------
    alpha : Array
        The concentration parameter of the Dirichlet distribution. Array of shape
        ``(n_arms, n_outcomes)`` where each row holds the per-arm posterior
        pseudo-counts over the categorical outcomes.
    """

    alpha: Array


class DiscreteThompsonSampling(BaseAgent):
    r"""
    Discrete Thompson sampling agent [12]_. The Dirichlet distribution is a conjugate prior for the categorical
    distribution over a finite set of outcomes. For each arm the agent maintains a Dirichlet posterior over the
    probabilities of the outcomes, which is updated after each observation. The expected reward of an arm is
    computed as the dot product between a probability vector sampled from the posterior and the outcome values,
    and the action with the highest expected value is selected.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms. :math:`N \in \mathbb{N}_{+}`.
    alpha : Array
        Initial concentration parameter of the Dirichlet prior, shared across arms. Array of shape
        ``(n_outcomes,)`` with strictly positive entries. See also ``DiscreteThompsonSamplingState`` for
        interpretation. :math:`\alpha_k > 0`.
    outcomes : Array
        Array of shape ``(n_outcomes,)`` containing the values of the possible rewards. The agent assumes that
        every observed reward exactly matches one of these values.

    References
    ----------
    .. [12] Stephen Tu. The Dirichlet-Multinomial and Dirichlet-Categorical models for Bayesian inference.
       https://stephentu.github.io/writeups/dirichlet-conjugate-prior.pdf
    """

    def __init__(
            self,
            n_arms: int,
            alpha: Array,
            outcomes: Array
    ) -> None:
        alpha = jnp.asarray(alpha)
        outcomes = jnp.asarray(outcomes)
        assert jnp.all(alpha > 0)

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=self.n_arms, alpha=alpha))
        self.update = jax.jit(partial(self.update, outcomes=outcomes))
        self.sample = jax.jit(partial(self.sample, outcomes=outcomes))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), int),
            'alpha': gym.spaces.Sequence(gym.spaces.Box(0.0, jnp.inf, (), float)),
            'outcomes': gym.spaces.Sequence(gym.spaces.Box(-jnp.inf, jnp.inf, (), float)),
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
            alpha: Array,
    ) -> DiscreteThompsonSamplingState:
        r"""
        Creates and initializes an instance of the discrete Thompson sampling agent for ``n_arms`` arms with the
        given Dirichlet prior over ``outcomes``. The same prior ``alpha`` is used for every arm.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of bandit arms.
        alpha : Array
            Initial Dirichlet concentration parameter of shape ``(n_outcomes,)``. See also
            ``DiscreteThompsonSamplingState`` for interpretation.

        Returns
        -------
        DiscreteThompsonSamplingState
            Initial state of the discrete Thompson sampling agent.
        """

        return DiscreteThompsonSamplingState(alpha=jnp.tile(alpha, (n_arms, 1)))

    @staticmethod
    def update(
            state: DiscreteThompsonSamplingState,
            key: PRNGKey,
            action: int,
            reward: Scalar,
            outcomes: Array
    ) -> DiscreteThompsonSamplingState:
        r"""
        Discrete Thompson sampling update according to [12]_. The observed reward is mapped to its categorical
        outcome and the corresponding concentration parameter for the selected arm is incremented by one:

        .. math::
          \alpha_{t + 1}(a, k) = \alpha_t(a, k) + \mathbb{1}\left[r_t = o_k\right]

        where :math:`o_k` is the :math:`k`-th element of ``outcomes``.

        Parameters
        ----------
        state : DiscreteThompsonSamplingState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        reward : Float
            Reward obtained upon execution of action.
        outcomes : Array
            Array of shape ``(n_outcomes,)`` with the possible reward values.

        Returns
        -------
        DiscreteThompsonSamplingState
            Updated agent state.
        """

        update = (reward == outcomes).astype(state.alpha.dtype)
        alpha = state.alpha.at[action].add(update)
        return DiscreteThompsonSamplingState(alpha=alpha)

    @staticmethod
    def sample(state: DiscreteThompsonSamplingState, key: PRNGKey, outcomes: Array) -> int:
        r"""
        The discrete Thompson sampling policy is stochastic. For each arm :math:`a` the algorithm draws a
        probability vector :math:`p_a \sim \operatorname{Dirichlet}(\alpha(a))` over the categorical outcomes and
        computes the expected reward as :math:`q_a = p_a \cdot o`, where :math:`o` is the vector of outcome
        values. The next action is selected as :math:`A = \operatorname*{argmax}_{a \in \mathscr{A}} q_a`,
        where :math:`\mathscr{A}` is a set of all actions.

        Parameters
        ----------
        state : DiscreteThompsonSamplingState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        outcomes : Array
            Array of shape ``(n_outcomes,)`` with the possible reward values.

        Returns
        -------
        int
            Selected action.
        """

        p = jax.random.dirichlet(key, state.alpha)
        expectation = p.dot(outcomes)

        return jnp.argmax(expectation)
