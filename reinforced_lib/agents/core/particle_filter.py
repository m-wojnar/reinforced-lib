from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from chex import dataclass, Array, Numeric, PRNGKey, Scalar, Shape

from reinforced_lib.agents import AgentState


@dataclass
class ParticleFilterState(AgentState):
    """
    Container for the state of the particle filter agent.

    Attributes
    ----------
    positions : array_like
        Positions of the particles.
    logit_weights : array_like
        Unnormalized log weights of the particles.
    """

    positions: Array
    logit_weights: Array


def simple_resample(operands: tuple[ParticleFilterState, PRNGKey]) -> ParticleFilterState:
    """
    Samples new particle positions from a categorical distribution with particle weights, then sets all weights equal.

    Parameters
    ----------
    operands : tuple[ParticleFilterState, PRNGKey]
        Tuple containing the filter state and a PRNG key.

    Returns
    -------
    ParticleFilterState
        Updated filter state.
    """

    state, key = operands
    positions_idx = jax.random.categorical(key, state.logit_weights, shape=state.positions.shape)

    return ParticleFilterState(
        positions=state.positions[positions_idx],
        logit_weights=jnp.zeros_like(state.logit_weights)
    )


def effective_sample_size(state: ParticleFilterState, threshold: Scalar = 0.5) -> bool:
    r"""
    Calculates the effective sample size [9]_ (ESS). If ESS is smaller than the number of sample times threshold,
    then a resampling is necessary.

    Parameters
    ----------
    state : ParticleFilterState
        Current state of the filter.
    threshold : float, default=0.5
        Threshold value used to decide if a resampling is necessary. :math:`thr \in (0, 1)`.

    Returns
    -------
    bool
        Information whether a resampling should be performed.

    References
    ----------
    .. [9] https://en.wikipedia.org/wiki/Effective_sample_size#Weighted_samples
    """

    weights = jax.nn.softmax(state.logit_weights)
    return 1 < jnp.sum(weights ** 2) * state.positions.size * threshold


def simple_transition(state: ParticleFilterState, key: PRNGKey, scale: Scalar, *args) -> ParticleFilterState:
    r"""
    Performs simple movement of the particle positions according to a normal distribution with
    :math:`\mu = 0` and :math:`\sigma = scale`.

    Parameters
    ----------
    state : ParticleFilterState
        Current state of the filter.
    key : PRNGKey
        A PRNG key used as the random key.
    scale : float
        Scale of the random movement of particles. :math:`scale > 0`.

    Returns
    -------
    ParticleFilterState
        Updated filter state.
    """

    positions_update = scale * jax.random.normal(key, state.positions.shape)

    return ParticleFilterState(
        positions=state.positions + positions_update,
        logit_weights=state.logit_weights
    )


def linear_transition(state: ParticleFilterState, key: PRNGKey, scale: Scalar, delta_time: Scalar) -> ParticleFilterState:
    r"""
    Performs movement of the particle positions according to a normal distribution with :math:`\mu = 0` and
    :math:`\sigma = scale \cdot \Delta t`, where :math:`\Delta t` is the time elapsed since the last update.

    Parameters
    ----------
    state : ParticleFilterState
        Current state of the filter.
    key : PRNGKey
        A PRNG key used as the random key.
    scale : float
        Scale of the random movement of particles. :math:`scale > 0`.
    delta_time : float
        Time elapsed since the last update.

    Returns
    -------
    ParticleFilterState
        Updated filter state.
    """

    return simple_transition(state, key, scale * delta_time)


def affine_transition(state: ParticleFilterState, key: PRNGKey, scale: Array, delta_time: Scalar) -> ParticleFilterState:
    r"""
    Performs movement of the particle positions according to a normal distribution with :math:`\mu = 0` and
    :math:`\sigma = scale_0 \cdot \Delta t + scale_1`, where :math:`\Delta t` is the time elapsed since
    the last update.

    Parameters
    ----------
    state : ParticleFilterState
        Current state of the filter.
    key : PRNGKey
        A PRNG key used as the random key.
    scale : array_like
        Scale of the random movement of particles. :math:`scale_0, scale_1 > 0`.
    delta_time : float
        Time elapsed since the last update.

    Returns
    -------
    ParticleFilterState
        Updated filter state.
    """

    return simple_transition(state, key, scale[0] * delta_time + scale[1])


class ParticleFilter:
    """
    Particle filter (sequential Monte Carlo) algorithm estimating the
    internal environment state given noisy or partial observations.

    Parameters
    ----------
    initial_distribution_fn : callable
        Function that samples the initial particle positions; takes two positional arguments:
            - ``key``: a PRNG key used as a random key (`PRNGKey`).
            - ``shape``: shape of the sample (`Shape`).

        Returns the initial particle positions (`Array`).

    positions_shape : array_like
        Shape of the particle positions array.
    weights_shape : array_like
        Shape of the particle weights array.
    scale : array_like
        Scale of the random movement of the particles.
    observation_fn : callable
        Function that updates particles based on an observation from the environment; takes two positional arguments:
            - ``state``: the state of the filter (`ParticleFilterState`).
            - ``observation``: an observation from the environment (`any`).
        
        Returns the updated state of the filter (`ParticleFilterState`).
    
    resample_fn : callable, default=particle_filter.simple_resample
        Function that performs resampling of the particles; takes one positional argument:
            - ``operands``: a tuple containing the filter state and a PRNG key (`tuple[ParticleFilterState, PRNGKey]`).
        
        Returns the updated state of the filter (`ParticleFilterState`).
    
    resample_criterion_fn : callable, default=particle_filter.effective_sample_size
        Function that checks if a resampling is necessary; takes one positional argument:
            - ``state``: the state of the filter (`ParticleFilterState`).
        
        Returns information whether a resampling should be performed (`bool`).
    
    transition_fn : callable, default=particle_filter.simple_transition
        Function that updates the particle positions; takes four positional arguments:
            - ``state``: the state of the filter (`ParticleFilterState`).
            - ``key``: a PRNG key used as a random key (`PRNGKey`).
            - ``scale``: scale of the random movement of the particles (`array_like`).
            - ``time``: the current time (`float`).
        
        Returns the updated state of the filter (`ParticleFilterState`).
    """

    def __init__(
            self,
            initial_distribution_fn: Callable,
            positions_shape: Shape,
            weights_shape: Shape,
            scale: Numeric,
            observation_fn: Callable[[ParticleFilterState, any], ParticleFilterState],
            resample_fn: Callable[[tuple[ParticleFilterState, PRNGKey]], ParticleFilterState] = simple_resample,
            resample_criterion_fn: Callable[[ParticleFilterState], bool] = effective_sample_size,
            transition_fn: Callable[[ParticleFilterState, PRNGKey, Numeric, Scalar], ParticleFilterState] = simple_transition
    ) -> None:
        self.init = jax.jit(
            partial(
                self.init,
                initial_distribution_fn=initial_distribution_fn,
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
            initial_distribution_fn: Callable,
            positions_shape: Shape,
            weights_shape: Shape
    ) -> ParticleFilterState:
        """
        Creates and initializes an instance of the particle filter.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        initial_distribution_fn : callable
            Function that samples the initial particle positions.
                - ``key``: PRNG key used as a random key (`PRNGKey`).
                - ``shape``: shape of the sample (`Shape`).

            Returns the initial particle positions (`Array`).

        positions_shape : array_like
            Shape of the particle positions array.
        weights_shape : array_like
            Shape of the particle weights array.

        Returns
        -------
        ParticleFilterState
            Initial state of the Particle Filter.
        """

        return ParticleFilterState(
            positions=initial_distribution_fn(key, positions_shape),
            logit_weights=jnp.zeros(weights_shape)
        )

    @staticmethod
    def update(
            state: ParticleFilterState,
            key: PRNGKey,
            observation_fn: Callable[[ParticleFilterState, any], ParticleFilterState],
            observation: any,
            resample_fn: Callable[[tuple[ParticleFilterState, PRNGKey]], ParticleFilterState],
            resample_criterion_fn: Callable[[ParticleFilterState], bool],
            transition_fn: Callable[[ParticleFilterState, PRNGKey, Numeric, Scalar], ParticleFilterState],
            delta_time: Scalar,
            scale: Numeric
    ) -> ParticleFilterState:
        """
        Updates the state of the filter based on an observation from the environment, then
        performs resampling (if necessary) and transition of the particles.

        Parameters
        ----------
        state : ParticleFilterState
            Current state of the filter.
        key : PRNGKey
            A PRNG key used as the random key.
        observation_fn : callable
            Function that updates particles based on an observation from the environment; takes two positional arguments:
                - ``state``: the state of the filter (`ParticleFilterState`).
                - ``observation``: an observation from the environment (`any`).

            Returns the updated state of the filter (`ParticleFilterState`).
        
        observation : any
            An observation from the environment.
        resample_fn : callable, default=particle_filter.simple_resample
            Function that performs resampling of the particles; takes one positional argument:
                - ``operands``: a tuple containing the filter state and a PRNG key (`tuple[ParticleFilterState, PRNGKey]`).

            Returns the updated state of the filter (`ParticleFilterState`).

        resample_criterion_fn : callable, default=particle_filter.effective_sample_size
            Function that checks if a resampling is necessary; takes one positional argument:
                - ``state``: the state of the filter (`ParticleFilterState`).

            Returns information whether a resampling should be performed (`bool`).

        transition_fn : callable, default=particle_filter.simple_transition
            Function that updates the particle positions; takes four positional arguments:
                - ``state``: the state of the filter (`ParticleFilterState`).
                - ``key``: a PRNG key used as a random key (`PRNGKey`).
                - ``scale``: scale of the random movement of the particles (`array_like`).
                - ``time``: the current time (`float`).

        delta_time : float
            Time difference between the current and the previous observation.
        scale : array_like
            Scale of the random movement of the particles.

        Returns
        -------
        ParticleFilterState
            Updated filter state.
        """

        resample_key, transition_key = jax.random.split(key)

        state = observation_fn(state, observation)
        state = jax.lax.cond(resample_criterion_fn(state), resample_fn, lambda op: op[0], (state, resample_key))
        state = transition_fn(state, transition_key, scale, delta_time)

        return ParticleFilterState(
            positions=state.positions,
            logit_weights=state.logit_weights
        )

    @staticmethod
    def sample(
            state: ParticleFilterState,
            key: PRNGKey
    ) -> Numeric:
        """
        Samples the estimated environment state from a categorical distribution with particle weights.

        Parameters
        ----------
        state : ParticleFilterState
            Current state of the filter.
        key : PRNGKey
            A PRNG key used as the random key.

        Returns
        -------
        array_like
            Estimated environment state.
        """

        return state.positions[jax.random.categorical(key, state.logit_weights)]
