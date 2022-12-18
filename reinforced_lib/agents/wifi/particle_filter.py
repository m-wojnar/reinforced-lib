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
    r"""
    Particle filter agent designed for the IEEE 802.11ax environments. The implementation is based on [2]_.
    The latent state of the filter is the channel condition described by the parameter :math:`\theta = \gamma − P_{tx}`
    (in logarithmic scale) where :math:`\gamma` is SINR and :math:`P_{tx}` is the current transmission power.

    Parameters
    ----------
    min_snr_init : float
        Minial SINR value [dBm] in the initial particle distribution.
    max_snr_init : float
        Maximal SINR value [dBm] in the initial particle distribution.
    default_power : float
        Default transmission power [dBm].
    particles_num : int, default=1000
        Number of created particles. :math:`N \in \mathbb{N}_{+}`.
    scale : float, default=10.0
        Velocity of a random movement of the particles. :math:`scale > 0`.

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
        Creates and initializes an instance of the particle filter agent with ``particles_num`` particles.
        Particle positions are dawn from a uniform distribution from ``min_snr_init`` to ``max_snr_init``,
        and particle weights are set equal.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.

        Returns
        -------
        ParticleFilterState
            Initial state of the particle filter agent.
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
        r"""
        The weight of the particle :math:`i` at step :math:`t + 1` is updated based on observation :math:`(r, s)`:

        .. math::
          p_{t + 1, i} =
          \begin{cases}
            p_{t, i} \cdot P(s | r, \gamma) (1 - P_c) & \text{if } s = 1 , \\
            p_{t, i} \cdot (1 - P(s | r, \gamma)) (1 - P_c) & \text{otherwise} ,
          \end{cases}

        where :math:`r` is the data rate used during the transmission, :math:`s` indicates if the transmission was
        successful, :math:`\gamma` is SINR, :math:`P(s | r, \gamma)` is the probability of transmission success
        conditioned by the data rate used and current SINR, and :math:`P_c` is the probability of unsuccessful
        transmission due to collision estimated as :math:`1 / CW` (contention window).

        Parameters
        ----------
        state : ParticleFilterState
            Current state of the agent.
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
            Contention window used during the transmission.
        scale : float
            Velocity of a random movement of particles.

        Returns
        -------
        ParticleFilterState
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
        r"""
        The algorithm draws :math:`\theta` from the categorical distribution according to the particles positions
        and weights. Then it calculates the corresponding SINR value :math:`\gamma = \theta + P_{tx}`.
        Next MCS is selected as:

        .. math::
          A = \operatorname*{argmax}_{i} P(1 | r_i, \gamma) r_i ,

        where :math:`r_i` is the data rate of MCS :math:`i`.

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
            Tuple containing the updated agent state and the selected action.
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
        Updates particles weights based on the observation from the environment.

        Parameters
        ----------
        state : ParticleFilterState
            Current state of the agent.
        observation : tuple
            Tuple containing ``action``, ``n_successful``, ``n_failed``, ``power``, and ``cw``.

        Returns
        -------
        ParticleFilterState
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
        Calculates an approximated probability of a successful transmission given the observed SINR value.

        Parameters
        ----------
        observed_snr : float
            Observed SINR value [dBm].

        Returns
        -------
        float
            Probability of a successful transmission for all MCS values.
        """

        return norm.cdf(observed_snr, loc=ParticleFilter._wifi_modes_snrs, scale=1 / jnp.sqrt(8))
