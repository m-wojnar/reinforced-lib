from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
from chex import dataclass, Array, Scalar, PRNGKey

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class EGreedyState(AgentState):
    r"""
    Container for the state of the :math:`\epsilon`-greedy agent.

    Attributes
    ----------
    Q : Array
        Action-value function estimates for each arm.
    N : Array
        Number of tries for each arm.
    e : float
        Experiment rate (epsilon).
    """

    Q: Array
    N: Array
    e: Scalar


class EGreedy(BaseAgent):
    r"""
    Epsilon-greedy [5]_ agent with an optimistic start behavior and optional exponential recency-weighted average update.
    It selects a random action from a set of all actions :math:`\mathscr{A}` with probability
    :math:`\epsilon` (exploration), otherwise it chooses the currently best action (exploitation).
    Epsilon can be decayed over time to shift the policy from exploration to exploitation.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms. :math:`N \in \mathbb{N}_{+}`.
    e : float
        Initial experiment rate (epsilon). :math:`\epsilon \in [0, 1]`.
    e_min : float, default=0.0
        Minimum value of the experiment rate. :math:`\epsilon_{\min} \in [0, 1]`.
    e_decay : float, default=1.0
        Decay factor for the experiment rate. :math:`\epsilon_{\text{decay}} \in [0, 1]`.
    optimistic_start : float, default=0.0
        Interpreted as the optimistic start to encourage exploration in the early stages.
    alpha : float, default=0.0
        If non-zero, exponential recency-weighted average is used to update :math:`Q` values. :math:`\alpha \in [0, 1]`.

    References
    ----------
    .. [5] Richard Sutton and Andrew Barto. 2018. Reinforcement Learning: An Introduction. The MIT Press.
    """

    def __init__(
            self,
            n_arms: int,
            e: Scalar,
            e_min: Scalar = 0.0,
            e_decay: Scalar = 1.0,
            optimistic_start: Scalar = 0.0,
            alpha: Scalar = 0.0
    ) -> None:
        assert 0 <= e <= 1
        assert 0 <= e_min <= 1
        assert 0 <= e_decay <= 1
        assert 0 <= alpha <= 1

        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=n_arms, e=e, optimistic_start=optimistic_start))
        self.update = jax.jit(partial(self.update, alpha=alpha, e_decay=e_decay, e_min=e_min))
        self.sample = jax.jit(self.sample)

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), int),
            'e': gym.spaces.Box(0.0, 1.0, (1,), float),
            'e_min': gym.spaces.Box(0.0, 1.0, (1,), float),
            'e_decay': gym.spaces.Box(0.0, 1.0, (1,), float),
            'optimistic_start': gym.spaces.Box(0.0, jnp.inf, (1,), float),
            'alpha': gym.spaces.Box(0.0, 1.0, (1,), float)
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
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.n_arms)

    @staticmethod
    def init(
            key: PRNGKey,
            n_arms: int,
            e: Scalar,
            optimistic_start: Scalar
    ) -> EGreedyState:
        r"""
        Creates and initializes instance of the :math:`\epsilon`-greedy agent for ``n_arms`` arms. Action-value function estimates are
        set to ``optimistic_start`` value and the number of tries is one for each arm.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of bandit arms.
        e : float
            Initial experiment rate (epsilon).
        optimistic_start : float
            Interpreted as the optimistic start to encourage exploration in the early stages.

        Returns
        -------
        EGreedyState
            Initial state of the :math:`\epsilon`-greedy agent.
        """

        return EGreedyState(
            Q=jnp.full((n_arms, 1), optimistic_start),
            N=jnp.ones((n_arms, 1), dtype=int),
            e=e
        )

    @staticmethod
    def update(
        state: EGreedyState,
        key: PRNGKey,
        action: int,
        reward: Scalar,
        alpha: Scalar,
        e_decay: Scalar,
        e_min: Scalar
    ) -> EGreedyState:
        r"""
        In the stationary case, the action-value estimate for a given arm is updated as
        :math:`Q_{t + 1} = Q_t + \frac{1}{t} \lbrack R_t - Q_t \rbrack` after receiving reward :math:`R_t` at step
        :math:`t` and the number of tries for the corresponding arm is incremented. In the non-stationary case,
        the update follows the equation :math:`Q_{t + 1} = Q_t + \alpha \lbrack R_t - Q_t \rbrack`.
        Exploration rate :math:`\epsilon` can be decayed over time following the equation
        :math:`\epsilon_{t + 1} = \max(\epsilon_{\min}, \epsilon_t \times \epsilon_{\text{decay}})`.

        Parameters
        ----------
        state : EGreedyState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        reward : float
            Reward collected by the agent after taking the previous action.
        alpha : float
            Exponential recency-weighted average factor (used when :math:`\alpha > 0`).
        e_decay : float
            Decay factor for the experiment rate.
        e_min : float
            Minimum value of the experiment

        Returns
        -------
        EGreedyState
            Updated agent state.
        """

        def classic_update(operands: tuple) -> EGreedyState:
            state, action, reward, alpha = operands
            return EGreedyState(
                Q=state.Q.at[action].add((reward - state.Q[action]) / state.N[action]),
                N=state.N.at[action].add(1),
                e=jnp.maximum(e_min, state.e * e_decay)
            )

        def erwa_update(operands: tuple) -> EGreedyState:
            state, action, reward, alpha = operands
            return EGreedyState(
                Q=state.Q.at[action].add(alpha * (reward - state.Q[action])),
                N=state.N.at[action].add(1),
                e=jnp.maximum(e_min, state.e * e_decay)
            )

        return jax.lax.cond(alpha == 0, classic_update, erwa_update, (state, action, reward, alpha))

    @staticmethod
    def sample(
        state: EGreedyState,
        key: PRNGKey
    ) -> int:
        r"""
        Epsilon-greedy agent follows the policy:

        .. math::
           A =
           \begin{cases}
              \operatorname*{argmax}_{a \in \mathscr{A}} Q(a) & \text{with probability } 1 - \epsilon , \\
              \text{random action} & \text{with probability } \epsilon . \\
           \end{cases}

        Parameters
        ----------
        state : EGreedyState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.

        Returns
        -------
        int
            Selected action.
        """

        max_q = (state.Q == state.Q.max()).astype(float)
        probs = (1 - state.e) * max_q / jnp.sum(max_q) + state.e / state.Q.shape[0]
        return jax.random.choice(key, state.Q.shape[0], p=probs.flatten())
