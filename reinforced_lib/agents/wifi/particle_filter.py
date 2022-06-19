from functools import partial
from typing import Tuple

import chex
import distrax
import gym.spaces
import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from reinforced_lib.agents.agent_state import AgentState
from reinforced_lib.agents.base_agent import BaseAgent


@chex.dataclass
class ParticleFilterState(AgentState):
    """
    Container for the state of the Particle Filter agent.

    Attributes
    ----------
    positions : array_like
        Positions of filter particles.
    weights : array_like
        Weights of filter particles.
    last_sample : float
        Time of the last sampling.
    """

    positions: chex.Array
    logit_weights: chex.Array
    last_sample: chex.Scalar


def init(
        min_position: chex.Scalar,
        max_position: chex.Scalar,
        particles_num: jnp.int32
) -> ParticleFilterState:
    """
    Creates and initializes instance of the Particle Filter agent.

    Parameters
    ----------
    min_position : float
        Minimum position of a particle.
    max_position : float
        Maximum position of a particle.
    particles_num : int
        Number of created particles.

    Returns
    -------
    state : ParticleFilterState
        Initial state of the Particle Filter agent.
    """

    return ParticleFilterState(
        positions=jnp.linspace(min_position, max_position, particles_num),
        logit_weights=jnp.zeros(particles_num),
        last_sample=0.0
    )


def update(
        state: ParticleFilterState,
        key: chex.PRNGKey,
        action: jnp.int32,
        n_successful: jnp.int32,
        n_failed: jnp.int32,
        time: chex.Scalar,
        power: chex.Scalar,
        cw: jnp.int32,
        min_snr: chex.Array,
        scale: chex.Scalar
) -> ParticleFilterState:
    """
    Updates the state of the agent after performing some action and receiving a reward.

    Parameters
    ----------
    state : ParticleFilterState
        Current state of agent.
    key : chex.PRNGKey
        A PRNG key used as the random key.
    action : int
        Previously selected action.
    n_successful : int
        Number of successful tries.
    n_failed : int
        Number of failed tries.
    time : float
        Current time.
    power : float
        Power used during the transmission.
    cw : int
        Contention Window used during the transmission.
    min_snr : array_like
        Minimal SNR value that is required for a successful transmission for each MCS.
    scale : float
        Velocity of the random movement of particles.

    Returns
    -------
    state : ParticleFilterState
        Updated agent state.
    """

    def on_success(operands: Tuple) -> ParticleFilterState:
        state, n, cw, p_s = operands
        return ParticleFilterState(
            positions=state.positions,
            logit_weights=state.logit_weights + n * jnp.log(p_s * (1 - 1 / cw)),
            last_sample=state.last_sample
        )

    def on_failure(operands: Tuple) -> ParticleFilterState:
        state, n, cw, p_s = operands
        return ParticleFilterState(
            positions=state.positions,
            logit_weights=state.logit_weights + n * jnp.log(1 - p_s * (1 - 1 / cw)),
            last_sample=state.last_sample
        )

    def resample(operands: Tuple) -> ParticleFilterState:
        state, logit_weights, key = operands
        positions_idx = distrax.Categorical(logit_weights).sample(seed=key, sample_shape=state.positions.size)
        return ParticleFilterState(
            positions=state.positions[positions_idx],
            logit_weights=jnp.zeros_like(logit_weights),
            last_sample=state.last_sample
        )

    particles_num = state.positions.size
    success_prob_func = jax.vmap(success_probability, in_axes=[None, 0])
    p_s = success_prob_func(min_snr[action], state.positions + power)

    state = jax.lax.cond(n_successful > 0, on_success, lambda op: op[0], (state, n_successful, cw, p_s))
    state = jax.lax.cond(n_failed > 0, on_failure, lambda op: op[0], (state, n_failed, cw, p_s))

    logit_weights = state.logit_weights - jnp.max(state.logit_weights)
    weights = jnp.exp(logit_weights)
    weights = weights / jnp.sum(weights)
    sample_size = 1 / jnp.sum(weights ** 2)
    state = jax.lax.cond(sample_size < particles_num / 2, resample, lambda op: op[0], (state, logit_weights, key))

    movement_scale = scale * (time - state.last_sample)
    positions = state.positions + distrax.Normal(0, movement_scale).sample(seed=key, sample_shape=particles_num)

    return ParticleFilterState(
        positions=positions,
        logit_weights=state.logit_weights,
        last_sample=state.last_sample
    )


def sample(
        state: ParticleFilterState,
        key: chex.PRNGKey,
        time: chex.Scalar,
        power: chex.Scalar,
        rates: chex.Array,
        min_snr: chex.Array
) -> Tuple[ParticleFilterState, jnp.int32]:
    """
    Selects next action based on current agent state.

    Parameters
    ----------
    state : ParticleFilterState
        Current state of the agent.
    key : chex.PRNGKey
        A PRNG key used as the random key.
    time : float
        Current time.
    power : float
        Power used during the transmission.
    rates : array_like
        Transmission data rates corresponding to each MCS.
    min_snr : array_like
        Minimal SNR value that is required for a successful transmission for each MCS.

    Returns
    -------
    tuple[ParticleFilterState, int]
        Tuple containing updated agent state and selected action.
    """
    
    success_prob_func = jax.vmap(success_probability, in_axes=[0, None])
    snr_sample = state.positions[distrax.Categorical(state.logit_weights).sample(seed=key)] + power
    p_s = success_prob_func(min_snr, snr_sample)
    
    action = jnp.argmax(p_s * rates)
    state = ParticleFilterState(
        positions=state.positions,
        logit_weights=state.logit_weights,
        last_sample=time
    )
    
    return state, action


def success_probability(min_snr: chex.Scalar, observed_snr: chex.Scalar) -> chex.Scalar:
    """
    Calculates approximated probability of a successful transmission for a given minimal and observed SNR.

    Parameters
    ----------
    min_snr : float
        Minimal SNR value that is required for a successful transmission.
    observed_snr : float
        Observed SNR value.

    Returns
    -------
    prob : float
        Probability of a successful transmission.
    """

    return 0.5 * (1 + erf(2 * (observed_snr - min_snr + 0.5)))


class ParticleFilter(BaseAgent):
    """
    Particle Filter agent designed for IEEE 802.11 environments. Implementation based on [1]_.

    Parameters
    ----------
    n_mcs : int
        Number of MCS modes.
    min_position : float
        Minimum position of a particle.
    max_position : float
        Maximum position of a particle.
    particles_num : int
        Number of created particles.
    scale : float, default=10.0
        Velocity of the random movement of particles.

    References
    ----------
    .. [1] Krotov, Alexander & Kiryanov, Anton & Khorov, Evgeny. (2020). Rate Control With Spatial Reuse
       for Wi-Fi 6 Dense Deployments. IEEE Access. 8. 168898-168909. 10.1109/ACCESS.2020.3023552.
    """

    def __init__(
            self,
            n_mcs: jnp.int32,
            min_position: chex.Scalar,
            max_position: chex.Scalar,
            particles_num: jnp.int32,
            scale: chex.Scalar = 10.0
    ) -> None:
        self.n_mcs = n_mcs

        self.init = jax.jit(partial(init, min_position=min_position, max_position=max_position, particles_num=particles_num))
        self.update = jax.jit(partial(update, scale=scale))
        self.sample = jax.jit(sample)

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_mcs),
            'n_successful': gym.spaces.Box(0, jnp.inf, (1,), jnp.int32),
            'n_failed': gym.spaces.Box(0, jnp.inf, (1,), jnp.int32),
            'time': gym.spaces.Box(0.0, jnp.inf, (1,)),
            'power': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'cw': gym.spaces.Discrete(32767),
            'min_snr': gym.spaces.Box(-jnp.inf, jnp.inf, (self.n_mcs,))
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'time': gym.spaces.Box(0.0, jnp.inf, (1,)),
            'power': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'rates': gym.spaces.Box(0.0, jnp.inf, (self.n_mcs,)),
            'min_snr': gym.spaces.Box(-jnp.inf, jnp.inf, (self.n_mcs,))
        })

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.n_mcs)
