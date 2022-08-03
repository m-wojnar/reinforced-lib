from functools import partial
from typing import Any, Callable, Tuple

import distrax
import jax
import jax.numpy as jnp
from chex import dataclass, Array, Numeric, PRNGKey, Scalar, Shape

from reinforced_lib.agents import AgentState


@dataclass
class ParticleFilterState(AgentState):
    """
    Container for the state of the Particle Filter.

    Attributes
    ----------
    positions : array_like
        Positions of filter particles.
    weights : array_like
        Weights of filter particles.
    last_measurement : float
        Time of the last update.
    """

    positions: Array
    logit_weights: Array
    last_measurement: Scalar


def simple_resample(operands: Tuple[ParticleFilterState, PRNGKey]) -> ParticleFilterState:
    """
    Samples new particles positions from categorical distribution based on weights.
    Sets all weights to be equal.

    Parameters
    ----------
    operands : tuple[ParticleFilterState, PRNGKey]
        Tuple containing filter state and a PRNG key.

    Returns
    -------
    state : ParticleFilterState
        Updated filter state.
    """

    state, key = operands
    logit_weights = state.logit_weights - jnp.max(state.logit_weights)
    positions_idx = distrax.Categorical(logit_weights).sample(seed=key, sample_shape=state.positions.shape)

    return ParticleFilterState(
        positions=state.positions[positions_idx],
        logit_weights=jnp.zeros_like(logit_weights),
        last_measurement=state.last_measurement
    )


def effective_sample_size(state: ParticleFilterState, threshold: Scalar = 0.5) -> bool:
    """
    Calculates effective sample size [3]_. If ESS is smaller than number of samples * threshold,
    than resampling is necessary.

    Parameters
    ----------
    state : ParticleFilterState
        Current state of the filter.
    threshold : float, default=0.5
        Threshold value used to decide if resampling is necessary.

    Returns
    -------
    perform_resampling : bool
        Information whether resampling should be performed.

    References
    ----------
    .. [3] https://en.wikipedia.org/wiki/Effective_sample_size#Weighted_samples
    """

    logit_weights = state.logit_weights - jnp.max(state.logit_weights)
    weights = jnp.exp(logit_weights)
    weights = weights / jnp.sum(weights)
    sample_size = 1 / jnp.sum(weights ** 2)

    return sample_size < state.positions.size * threshold


def simple_transition(state: ParticleFilterState, key: PRNGKey, scale: Scalar, *args) -> ParticleFilterState:
    """
    Performs simple movement of particles positions based on normal distribution with
    ``mean = 0`` and ``standard deviation = scale``.

    Parameters
    ----------
    state : ParticleFilterState
        Current state of the filter.
    key : PRNGKey
        A PRNG key used as the random key.
    scale : float
        Scale of the random movement of particles.

    Returns
    -------
    state : ParticleFilterState
        Updated filter state.
    """

    positions_update = distrax.Normal(0, scale).sample(sample_shape=state.positions.shape, seed=key)

    return ParticleFilterState(
        positions=state.positions + positions_update,
        logit_weights=state.logit_weights,
        last_measurement=state.last_measurement
    )


def linear_transition(state: ParticleFilterState, key: PRNGKey, scale: Scalar, time: Scalar) -> ParticleFilterState:
    """
    Performs movement of particles positions based on normal distribution with
    ``mean = 0`` and ``standard deviation = scale * (last_measurement - time)``.

    Parameters
    ----------
    state : ParticleFilterState
        Current state of the filter.
    key : PRNGKey
        A PRNG key used as the random key.
    scale : float
        Scale of the random movement of particles.
    time : float
        Current time.

    Returns
    -------
    state : ParticleFilterState
        Updated filter state.
    """

    return simple_transition(state, key, scale * (state.last_measurement - time))


def affine_transition(state: ParticleFilterState, key: PRNGKey, scale: Array, time: Scalar) -> ParticleFilterState:
    """
    Performs movement of particles positions based on normal distribution with
    ``mean = 0`` and ``standard deviation = scale_a * (last_measurement - time) + scale_b``.

    Parameters
    ----------
    state : ParticleFilterState
        Current state of the filter.
    key : PRNGKey
        A PRNG key used as the random key.
    scale : array_like
        Scale of the random movement of particles.
    time : float
        Current time.

    Returns
    -------
    state : ParticleFilterState
        Updated filter state.
    """

    return simple_transition(state, key, scale[0] * (state.last_measurement - time) + scale[1])


class ParticleFilter:
    """
    Particle Filter (Sequential Monte Carlo) algorithm estimating
    internal environment state given noisy or partial observations.

    Parameters
    ----------
    initial_distribution : distrax.Distribution
        Distribution from which the initial state is sampled.
    positions_shape : array_like
        Shape of the particles positions array.
    weights_shape : array_like
        Shape of the particles weights array.
    scale : float or array_like
        Scale of the random movement of particles.
    observation_fn : callable
        Function that updates particles based on the observation of the environment, takes two positional arguments:
            - ``state``: state of the filter (`ParticleFilterState`).
            - ``observation``: observation of the environment (`any`).
        
        Returns updated state of the filter (`ParticleFilterState`).
    
    resample_fn : callable, default=particle_filter.simple_resample
        Function that performs resampling of particles, takes one positional argument:
            - ``operands``: tuple containing filter state and a PRNG key (`tuple[ParticleFilterState, PRNGKey]`).
        
        Returns updated state of the filter (`ParticleFilterState`).
    
    resample_criterion_fn : callable, default=particle_filter.effective_sample_size
        Function that checks if resampling is necessary, takes one positional argument:
            - ``state``: state of the filter (`ParticleFilterState`).
        
        Returns information whether resampling should be performed (`bool`).
    
    transition_fn : callable, default=particle_filter.simple_transition
        Function that updates particles positions, takes four positional arguments:
            - ``state``: state of the filter (`ParticleFilterState`).
            - ``key``: a PRNG key used as the random key (`PRNGKey`).
            - ``scale``: scale of the random movement of particles (`float or array_like`).
            - ``time``: current time (`float`).
        
        Returns updated state of the filter (`ParticleFilterState`).
    """

    def __init__(
            self,
            initial_distribution: distrax.Distribution,
            positions_shape: Shape,
            weights_shape: Shape,
            scale: Numeric,
            observation_fn: Callable[[ParticleFilterState, Any], ParticleFilterState],
            resample_fn: Callable[[Tuple[ParticleFilterState, PRNGKey]], ParticleFilterState] = simple_resample,
            resample_criterion_fn: Callable[[ParticleFilterState], bool] = effective_sample_size,
            transition_fn: Callable[[ParticleFilterState, PRNGKey, Numeric, Scalar], ParticleFilterState] = simple_transition
    ) -> None:
        self.init = jax.jit(
            partial(
                self.init,
                initial_distribution=initial_distribution,
                positions_shape=positions_shape,
                weights_shape=weights_shape
            )
        )
        self.update = jax.jit(
            partial(
                self.update,
                observation_fn=observation_fn,
                resample_fn=resample_fn,
                resample_criterion_fn=resample_criterion_fn,
                transition_fn=transition_fn,
                scale=scale
            )
        )
        self.sample = jax.jit(self.sample)

    @staticmethod
    def init(
            key: PRNGKey,
            initial_distribution: distrax.Distribution,
            positions_shape: Shape,
            weights_shape: Shape
    ) -> ParticleFilterState:
        """
        Creates and initializes instance of the Particle Filter.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        initial_distribution : distrax.Distribution
            Distribution from which the initial state is sampled.
        positions_shape : array_like
            Shape of the particles positions array.
        weights_shape : array_like
            Shape of the particles weights array.

        Returns
        -------
        state : ParticleFilterState
            Initial state of the Particle Filter.
        """

        return ParticleFilterState(
            positions=initial_distribution.sample(sample_shape=positions_shape, seed=key),
            logit_weights=jnp.zeros(weights_shape),
            last_measurement=0.0
        )

    @staticmethod
    def update(
            state: ParticleFilterState,
            key: PRNGKey,
            observation_fn: Callable[[ParticleFilterState, Any], ParticleFilterState],
            observation: Any,
            resample_fn: Callable[[Tuple[ParticleFilterState, PRNGKey]], ParticleFilterState],
            resample_criterion_fn: Callable[[ParticleFilterState], bool],
            transition_fn: Callable[[ParticleFilterState, PRNGKey, Numeric, Scalar], ParticleFilterState],
            time: Scalar,
            measurement_time: Scalar,
            scale: Numeric
    ) -> ParticleFilterState:
        """
        Updates the state of the filter based on the observation of the environment,
        performs resampling (if necessary) and transition of particles.

        Parameters
        ----------
        state : ParticleFilterState
            Current state of the filter.
        key : PRNGKey
            A PRNG key used as the random key.
        observation_fn : callable
            Function that updates particles based on the observation of the environment, takes two positional arguments:
                - ``state``: state of the filter (`ParticleFilterState`).
                - ``observation``: observation of the environment (`any`).
            
            Returns updated state of the filter (`ParticleFilterState`).
        
        observation : any
            Observation of the environment.
        resample_fn : callable
            Function that performs resampling of particles, takes one positional argument:
                - ``operands``: tuple containing filter state and a PRNG key (`tuple[ParticleFilterState, PRNGKey]`).
            
            Returns updated state of the filter (`ParticleFilterState`).
        
        resample_criterion_fn : callable
            Function that checks if resampling is necessary, takes one positional argument:
                - ``state``: state of the filter (`ParticleFilterState`).
            
            Returns information whether resampling should be performed (`bool`).

        transition_fn : callable
            Function that updates particles positions, takes four positional arguments:
                - ``state``: state of the filter (`ParticleFilterState`).
                - ``key``: a PRNG key used as the random key (`PRNGKey`).
                - ``scale``: scale of the random movement of particles (`float or array_like`).
                - ``time``: current time (`float`).
            
            Returns updated state of the filter (`ParticleFilterState`).

        time : float
            Current time.
        measurement_time : float
            Last measurement time.
        scale : float or array_like
            Scale of the random movement of particles.

        Returns
        -------
        state : ParticleFilterState
            Updated filter state.
        """

        resample_key, transition_key = jax.random.split(key)

        state = observation_fn(state, observation)
        state = jax.lax.cond(resample_criterion_fn(state), resample_fn, lambda op: op[0], (state, resample_key))
        state = transition_fn(state, transition_key, scale, time)

        return ParticleFilterState(
            positions=state.positions,
            logit_weights=state.logit_weights,
            last_measurement=measurement_time
        )

    @staticmethod
    def sample(
            state: ParticleFilterState,
            key: PRNGKey
    ) -> Tuple[ParticleFilterState, Numeric]:
        """
        Samples estimated environment state based on the current filter state.

        Parameters
        ----------
        state : ParticleFilterState
            Current state of the filter.
        key : PRNGKey
            A PRNG key used as the random key.

        Returns
        -------
        tuple[ParticleFilterState, float or array_like]
            Tuple containing filter state and estimated state.
        """

        return state, state.positions[distrax.Categorical(state.logit_weights).sample(seed=key)]
