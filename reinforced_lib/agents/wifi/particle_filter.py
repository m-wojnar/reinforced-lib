from functools import partial
from typing import Tuple

import gym.spaces
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, Scalar
from jax.scipy.stats import norm

from reinforced_lib.agents import BaseAgent
from reinforced_lib.agents.core.particle_filter import ParticleFilter as ParticleFilterBase
from reinforced_lib.agents.core.particle_filter import ParticleFilterState, linear_transition


class ParticleFilter(BaseAgent):
    """
    Particle Filter agent designed for IEEE 802.11ax environments. Implementation based on [2]_.

    Parameters
    ----------
    min_snr_init : float
        Minial SNR value [dBm] in initial particles' distribution.
    max_snr_init : float
        Maximal SNR value [dBm] in initial particles' distribution.
    default_power : float
        Default transmission power [dBm].
    particles_num : int, default=1000
        Number of created particles.
    scale : float, default=10.0
        Velocity of the random movement of particles.

    References
    ----------
    .. [2] Krotov, Alexander & Kiryanov, Anton & Khorov, Evgeny. (2020). Rate Control With Spatial Reuse
       for Wi-Fi 6 Dense Deployments. IEEE Access. 8. 168898-168909. 10.1109/ACCESS.2020.3023552.
    """

    _wifi_modes_snrs = jnp.array([
        4.12, 7.15, 10.05, 13.66,
        16.81, 21.58, 22.80, 23.96,
        28.66, 29.82, 33.82, 35.30
    ])

    def __init__(
            self,
            default_power: Scalar,
            min_snr_init: Scalar = 0.0,
            max_snr_init: Scalar = 40.0,
            particles_num: jnp.int32 = 1000,
            scale: Scalar = 10.0
    ) -> None:
        assert scale > 0
        assert particles_num > 0

        self.n_mcs = len(ParticleFilter._wifi_modes_snrs)

        self.pf = ParticleFilterBase(
            initial_distribution_fn=lambda key, shape: 
                jax.random.uniform(key, shape, minval=min_snr_init, maxval=max_snr_init) - default_power,
            positions_shape=(particles_num,),
            weights_shape=(particles_num,),
            scale=scale,
            observation_fn=jax.jit(self._observation_fn),
            transition_fn=linear_transition
        )

        self.update = partial(self.update, scale=scale)
        self.sample = jax.jit(partial(self.sample, pf=self.pf))

    @staticmethod
    def parameters_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'default_power': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'min_snr_init': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'max_snr_init': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'particles_num': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'scale': gym.spaces.Box(0.0, jnp.inf, (1,))
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_mcs),
            'n_successful': gym.spaces.Box(0, jnp.inf, (1,), jnp.int32),
            'n_failed': gym.spaces.Box(0, jnp.inf, (1,), jnp.int32),
            'time': gym.spaces.Box(0.0, jnp.inf, (1,)),
            'power': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'cw': gym.spaces.Discrete(32767)
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'time': gym.spaces.Box(0.0, jnp.inf, (1,)),
            'power': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'rates': gym.spaces.Box(0.0, jnp.inf, (self.n_mcs,))
        })

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.n_mcs)

    def init(self, key: PRNGKey) -> ParticleFilterState:
        """
        Creates and initializes instance of the Particle Filter agent.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.

        Returns
        -------
        state : ParticleFilterState
            Initial state of the Particle Filter agent.
        """

        return self.pf.init(key)

    def update(
            self,
            state: ParticleFilterState,
            key: PRNGKey,
            action: jnp.int32,
            n_successful: jnp.int32,
            n_failed: jnp.int32,
            time: Scalar,
            power: Scalar,
            cw: jnp.int32,
            scale: Scalar
    ) -> ParticleFilterState:
        """
        Updates the state of the agent after performing some action and receiving a reward.

        Parameters
        ----------
        state : ParticleFilterState
            Current state of agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        n_successful : int
            Number of successful tries.
        n_failed : int
            Number of failed tries.
        time : float
            Current time [s].
        power : float
            Power used during the transmission [dBm].
        cw : int
            Contention Window used during the transmission.
        scale : float
            Velocity of the random movement of particles.

        Returns
        -------
        state : ParticleFilterState
            Updated agent state.
        """

        return self.pf.update(
            state=state,
            key=key,
            observation=(action, n_successful, n_failed, power, cw),
            time=time,
            scale=scale
        )

    @staticmethod
    def sample(
            state: ParticleFilterState,
            key: PRNGKey,
            time: Scalar,
            power: Scalar,
            rates: Array,
            pf: ParticleFilterBase
    ) -> Tuple[ParticleFilterState, jnp.int32]:
        """
        Selects next action based on current agent state.

        Parameters
        ----------
        state : ParticleFilterState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        time : float
            Current time [s].
        power : float
            Power used during the transmission [dBm].
        rates : array_like
            Transmission data rates corresponding to each MCS [Mb/s].
        pf : ParticleFilterBase
            Instance of the base ParticleFilter class.

        Returns
        -------
        tuple[ParticleFilterState, int]
            Tuple containing updated agent state and selected action.
        """

        _, snr_sample = pf.sample(state, key)
        p_s = ParticleFilter._success_probability(snr_sample + power)

        return state, jnp.argmax(p_s * rates)

    @staticmethod
    def _observation_fn(
            state: ParticleFilterState,
            observation: Tuple[jnp.int32, jnp.int32, jnp.int32, Scalar, jnp.int32, Array]
    ) -> ParticleFilterState:
        """
        Updates particles weights based on the observation of the environment.

        Parameters
        ----------
        state : ParticleFilterState
            Current state of the agent.
        observation : tuple
            Tuple containing ``action``, ``n_successful``, ``n_failed``, ``power``, and ``cw``.

        Returns
        -------
        state : ParticleFilterState
            Updated state of the agent.
        """

        action, n_successful, n_failed, power, cw = observation
        p_s = jax.vmap(ParticleFilter._success_probability)(state.positions + power)[:, action]

        weights_update = jnp.where(n_successful > 0, n_successful * jnp.log(p_s * (1 - 1 / cw)), 0) + \
                         jnp.where(n_failed > 0, n_failed * jnp.log(1 - p_s * (1 - 1 / cw)), 0)
        logit_weights = state.logit_weights + weights_update

        return ParticleFilterState(
            positions=state.positions,
            logit_weights=logit_weights - jnp.nanmax(logit_weights),
            last_measurement=state.last_measurement
        )

    @staticmethod
    def _success_probability(observed_snr: Scalar) -> Array:
        """
        Calculates approximated probability of a successful transmission for a given minimal and observed SNR.

        Parameters
        ----------
        observed_snr : float
            Observed SNR value [dBm].

        Returns
        -------
        prob : float
            Probability of a successful transmission for all MCS values.
        """

        return norm.cdf(observed_snr, loc=ParticleFilter._wifi_modes_snrs, scale=1 / jnp.sqrt(8))
