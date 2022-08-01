from functools import partial
from typing import Tuple

import distrax
import gym.spaces
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, Scalar
from jax.scipy.special import erf

from reinforced_lib.agents import BaseAgent
from reinforced_lib.agents.core.particle_filter import ParticleFilter as ParticleFilterBase
from reinforced_lib.agents.core.particle_filter import ParticleFilterState, linear_transition


class ParticleFilter(BaseAgent):
    """
    Particle Filter agent designed for IEEE 802.11 environments. Implementation based on [2]_.

    Parameters
    ----------
    n_mcs : int
        Number of MCS modes.
    min_snr : float
        Minial approximated SNR value [dBm].
    max_snr : float
        Maximal approximated SNR value [dBm].
    initial_power : float
        Initial transmission power [dBm].
    particles_num : int, default=1000
        Number of created particles.
    scale : float, default=10.0
        Velocity of the random movement of particles.

    References
    ----------
    .. [2] Krotov, Alexander & Kiryanov, Anton & Khorov, Evgeny. (2020). Rate Control With Spatial Reuse
       for Wi-Fi 6 Dense Deployments. IEEE Access. 8. 168898-168909. 10.1109/ACCESS.2020.3023552.
    """

    def __init__(
            self,
            n_mcs: jnp.int32,
            min_snr: Scalar,
            max_snr: Scalar,
            initial_power: Scalar,
            particles_num: jnp.int32 = 1000,
            scale: Scalar = 10.0
    ) -> None:
        self.n_mcs = n_mcs

        self.pf = ParticleFilterBase(
            initial_distribution=distrax.Uniform(min_snr - initial_power, max_snr - initial_power),
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
            'n_mcs': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'min_snr': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'max_snr': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'initial_power': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
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
            'cw': gym.spaces.Discrete(32767),
            'min_snrs': gym.spaces.Box(-jnp.inf, jnp.inf, (self.n_mcs,))
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'time': gym.spaces.Box(0.0, jnp.inf, (1,)),
            'power': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'rates': gym.spaces.Box(0.0, jnp.inf, (self.n_mcs,)),
            'min_snrs': gym.spaces.Box(-jnp.inf, jnp.inf, (self.n_mcs,))
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
            scale: Scalar,
            min_snrs: Array
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
        min_snrs : array_like
            Minimal SNR value that is required for a successful transmission for each MCS [dBm].

        Returns
        -------
        state : ParticleFilterState
            Updated agent state.
        """

        return self.pf.update(
            state=state,
            key=key,
            observation=(action, n_successful, n_failed, power, cw, min_snrs),
            time=time,
            measurement_time=time,
            scale=scale
        )

    @staticmethod
    def sample(
            state: ParticleFilterState,
            key: PRNGKey,
            time: Scalar,
            power: Scalar,
            rates: Array,
            min_snrs: Array,
            pf: ParticleFilterBase,
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
        min_snrs : array_like
            Minimal SNR value that is required for a successful transmission for each MCS [dBm].
        pf : ParticleFilterBase
            Instance of the base ParticleFilter class.

        Returns
        -------
        tuple[ParticleFilterState, int]
            Tuple containing updated agent state and selected action.
        """

        success_probability_fn = jax.vmap(ParticleFilter._success_probability, in_axes=[0, None])

        _, snr_sample = pf.sample(state, key)
        p_s = success_probability_fn(min_snrs, snr_sample + power)

        action = jnp.argmax(p_s * rates)
        state = ParticleFilterState(
            positions=state.positions,
            logit_weights=state.logit_weights,
            last_measurement=time
        )

        return state, action

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
            Tuple containing action, n_successful, n_failed, power, cw, and min_snrs.

        Returns
        -------
        state : ParticleFilterState
            Updated state of the agent.
        """

        def on_success(operands: Tuple) -> ParticleFilterState:
            state, n, cw, p_s = operands
            return ParticleFilterState(
                positions=state.positions,
                logit_weights=state.logit_weights + n * jnp.log(p_s * (1 - 1 / cw)),
                last_measurement=state.last_measurement
            )

        def on_failure(operands: Tuple) -> ParticleFilterState:
            state, n, cw, p_s = operands
            return ParticleFilterState(
                positions=state.positions,
                logit_weights=state.logit_weights + n * jnp.log(1 - p_s * (1 - 1 / cw)),
                last_measurement=state.last_measurement
            )

        success_probability_fn = jax.vmap(ParticleFilter._success_probability, in_axes=[None, 0])

        action, n_successful, n_failed, power, cw, min_snrs = observation
        p_s = success_probability_fn(min_snrs[action], state.positions + power)

        state = jax.lax.cond(n_successful > 0, on_success, lambda op: op[0], (state, n_successful, cw, p_s))
        state = jax.lax.cond(n_failed > 0, on_failure, lambda op: op[0], (state, n_failed, cw, p_s))

        return state

    @staticmethod
    def _success_probability(min_snr: Scalar, observed_snr: Scalar) -> Scalar:
        """
        Calculates approximated probability of a successful transmission for a given minimal and observed SNR.

        Parameters
        ----------
        min_snr : float
            Minimal SNR value that is required for a successful transmission [dBm].
        observed_snr : float
            Observed SNR value [dBm].

        Returns
        -------
        prob : float
            Probability of a successful transmission.
        """

        return 0.5 * (1 + erf(2 * (observed_snr - min_snr + 0.5)))
