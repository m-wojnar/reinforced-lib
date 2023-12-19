import jax
import jax.numpy as jnp
from chex import PRNGKey, Scalar

from reinforced_lib.agents.mab.normal_thompson_sampling import NormalThompsonSampling, NormalThompsonSamplingState


class LogNormalThompsonSampling(NormalThompsonSampling):
    r"""
    Log-normal Thompson sampling agent. This algorithm is designed to handle positive rewards by transforming
    them into the log-space. For more details, refer to the documentation on ``NormalThompsonSampling``.
    """

    @staticmethod
    def update(
            state: NormalThompsonSamplingState,
            key: PRNGKey,
            action: int,
            reward: Scalar
    ) -> NormalThompsonSamplingState:
        r"""
        Log-normal Thompson sampling update. The update is analogous to the one in ``NormalThompsonSampling`` except
        that the reward is transformed into the log-space.

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

        return NormalThompsonSampling.update(state, key, action, jnp.log(reward))

    @staticmethod
    def sample(state: NormalThompsonSamplingState, key: PRNGKey) -> int:
        r"""
        Sampling actions is analogous to the one in ``NormalThompsonSampling`` except that the expected value of
        the log-normal distribution is computed instead of the expected value of the normal distribution.

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
        log_normal_mean = jnp.exp(loc + 0.5 * jnp.square(scale))

        return jnp.argmax(log_normal_mean)
