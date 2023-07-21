from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from chex import dataclass, Array, Scalar, PRNGKey

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class SoftmaxState(AgentState):
    r"""
    Container for the state of the Softmax agent.

    Attributes
    ----------
    H : array_like
        Preference for each arm.
    r : float
        Average of all obtained rewards :math:`\bar{R}`.
    n : int
        Step number.
    """

    H: Array
    r: Scalar
    n: jnp.int64


class Softmax(BaseAgent):
    r"""
    Softmax agent with baseline and optional exponential recency-weighted average update. It learns a preference
    function :math:`H`, which indicates a preference of selecting one arm over others. Algorithm policy can be
    controlled by the temperature parameter :math:`\tau`. The implementation is inspired by the work of Sutton and Barto [5]_.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms. :math:`N \in \mathbb{N}_{+}`.
    lr : float
        Step size. :math:`lr > 0`.
    alpha : float, default=0.0
        If non-zero, exponential recency-weighted average is used to update :math:`\bar{R}`. :math:`\alpha \in [0, 1]`.
    tau : float, default=1.0
        Temperature parameter. :math:`\tau > 0`.
    multiplier : float, default=1.0
        Multiplier for the reward. :math:`multiplier > 0`.
    """

    def __init__(
            self,
            n_arms: jnp.int32,
            lr: Scalar,
            alpha: Scalar = 0.0,
            tau: Scalar = 1.0,
            multiplier: Scalar = 1.0
    ) -> None:
        assert lr > 0
        assert 0 <= alpha <= 1
        assert tau > 0
        assert multiplier > 0

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=n_arms))
        self.update = jax.jit(partial(self.update, lr=lr, alpha=alpha, tau=tau, multiplier=multiplier))
        self.sample = jax.jit(partial(self.sample, tau=tau))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'lr': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32),
            'alpha': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32),
            'tau': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32)
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
    ) -> SoftmaxState:
        r"""
        Creates and initializes instance of the Softmax agent for ``n_arms`` arms. Preferences :math:`H` for each arm
        are set to zero, as well as the average of all rewards :math:`\bar{R}`. The step number :math:`n` is
        initialized to one.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of bandit arms.

        Returns
        -------
        SoftmaxState
            Initial state of the Softmax agent.
        """

        return SoftmaxState(
            H=jnp.zeros((n_arms, 1)),
            r=0.0,
            n=1
        )

    @staticmethod
    def update(
        state: SoftmaxState,
        key: PRNGKey,
        action: jnp.int32,
        reward: Scalar,
        lr: Scalar,
        alpha: Scalar,
        tau: Scalar,
        multiplier: Scalar
    ) -> SoftmaxState:
        r"""
        Preferences :math:`H` can be learned by stochastic gradient ascent. The softmax algorithm searches
        for such a set of preferences that maximizes the expected reward :math:`\mathbb{E}[R]`.
        The updates of :math:`H` for each action :math:`a` are calculated as:

        .. math::
           H_{t + 1}(a) = H_t(a) + \alpha (R_t - \bar{R}_t)(\mathbb{1}_{A_t = a} - \pi_t(a)),

        where :math:`\bar{R_t}` is the average of all rewards up to but not including step :math:`t`
        (by definition :math:`\bar{R}_1 = R_1`). The derivation of given formula can be found in [5]_.

        In the stationary case, :math:`\bar{R_t}` can be calculated as
        :math:`\bar{R}_{t + 1} = \bar{R}_t + \frac{1}{t} \lbrack R_t - \bar{R}_t \rbrack`. To improve the
        algorithm's performance in the non-stationary case, we apply
        :math:`\bar{R}_{t + 1} = \bar{R}_t + \alpha \lbrack R_t - \bar{R}_t \rbrack` with a constant
        step size :math:`\alpha`.

        Reward :math:`R_t` is multiplied by ``multiplier`` before updating preferences to allow for
        more flexible reward scaling while keeping the algorithm's properties.

        Parameters
        ----------
        state : SoftmaxState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        reward : float
            Reward collected by the agent after taking the previous action.
        lr : float
            Step size.
        alpha : float
            Exponential recency-weighted average factor (used when :math:`\alpha > 0`).
        tau : float
            Temperature parameter.
        multiplier : float
            Multiplier for the reward.

        Returns
        -------
        SoftmaxState
            Updated agent state.
        """

        reward *= multiplier

        r = jnp.where(state.n == 1, reward, state.r)
        pi = jax.nn.softmax(state.H / tau)

        return SoftmaxState(
            H=state.H + lr * (reward - r) * (jnp.zeros_like(state.H).at[action].set(1) - pi),
            r=r + (reward - r) * jnp.where(alpha == 0, 1 / state.n, alpha),
            n=state.n + 1
        )

    @staticmethod
    def sample(
        state: SoftmaxState,
        key: PRNGKey,
        tau: Scalar
    ) -> jnp.int32:
        r"""
        The policy of the Softmax algorithm is stochastic. The algorithm draws the next action from the softmax
        distribution. The probability of selecting action :math:`i` is calculated as:

        .. math::
           softmax(H)_i = \frac{\exp(H_i / \tau)}{\sum_{h \in H} \exp(h / \tau)} .

        Parameters
        ----------
        state : SoftmaxState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        tau : float
            Temperature parameter.

        Returns
        -------
        int
            Selected action.
        """

        return jax.random.categorical(key, state.H.squeeze() / tau)
