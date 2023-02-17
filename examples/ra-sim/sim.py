import random
import sys
from typing import Callable, Dict, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
from chex import dataclass, Array, Numeric, PRNGKey, Scalar
from jax.scipy.stats import norm


gym.envs.registration.register(
    id='RASimEnv-v1',
    entry_point='examples.ra-sim.sim:RASimEnv'
)


# CW constraints
MIN_CW_EXP = 4
MAX_CW_EXP = 10

# mean value based on the ns-3 static simulation with 1 station, ideal channel, AMPDU on, and constant MCS
FRAMES_PER_SECOND = 188

# based on the ns-3 static simulation with 1 station, ideal channel, and constant MCS
AMPDU_SIZES = jnp.array([3, 6, 9, 12, 18, 25, 28, 31, 37, 41, 41, 41])

# default values used by the ns-3 simulator
# https://www.nsnam.org/docs/models/html/wifi-testing.html#packet-error-rate-performance
DEFAULT_NOISE = -93.97
DEFAULT_TX_POWER = 16.0206

# log-distance channel model
# https://www.nsnam.org/docs/models/html/propagation.html#logdistancepropagationlossmodel
REFERENCE_SNR = DEFAULT_TX_POWER - DEFAULT_NOISE
REFERENCE_LOSS = 46.6777
EXPONENT = 3.0

# empirically selected values to match the ns-3 simulations results
MIN_SNRS = jnp.array([12.10, 12.10, 12.10, 12.10, 12.25, 16.25, 17.50, 19.00, 22.80, 24.50, 30.75, 32.75])


def distance_to_snr(distance: Numeric) -> Numeric:
    return REFERENCE_SNR - (REFERENCE_LOSS + 10 * EXPONENT * jnp.log10(distance))


def success_probability(snr: Scalar, mcs: jnp.int32) -> Scalar:
    return norm.cdf(snr, loc=MIN_SNRS[mcs], scale=1 / jnp.sqrt(8))


def collision_probability(n_wifi: jnp.int32) -> Scalar:
    return 0.154887 * jnp.log(1.03102 * n_wifi)


@dataclass
class RASimState:
    time: Array
    snr: Array
    ptr: jnp.int32
    cw: jnp.int32


@dataclass
class Env:
    init: Callable
    step: Callable


def ra_sim(
        simulation_time: Scalar,
        velocity: Scalar,
        initial_position: Scalar,
        n_wifi: jnp.int32,
        total_frames: jnp.int32
) -> Env:

    phy_rates = jnp.array([6.8, 13.6, 20.4, 27.2, 40.8, 54.6, 61.3, 68.1, 81.8, 90.9, 101.8, 112.5])

    @jax.jit
    def init() -> Tuple[RASimState, Dict]:
        distance = jnp.abs(jnp.linspace(0.0, simulation_time * velocity, total_frames) + initial_position)

        state = RASimState(
            time=jnp.linspace(0.0, simulation_time, total_frames),
            snr=distance_to_snr(distance),
            ptr=0,
            cw=MIN_CW_EXP
        )

        return state, _get_env_state(state, 0, 0, 0)

    def _get_env_state(state: RASimState, action: jnp.int32, n_successful: jnp.int32, n_failed: jnp.int32) -> Dict:
        return {
            'time': state.time[state.ptr],
            'n_successful': n_successful,
            'n_failed': n_failed,
            'n_wifi': n_wifi,
            'power': DEFAULT_TX_POWER,
            'cw': 2 ** state.cw - 1,
            'mcs': action,
            'terminal': False
        }

    @jax.jit
    def step(state: RASimState, action: jnp.int32, key: PRNGKey) -> Tuple[RASimState, Dict, Scalar, jnp.bool_]:
        n_all = AMPDU_SIZES[action]
        n_successful = (n_all * success_probability(state.snr[state.ptr], action)).astype(jnp.int32)
        collision = collision_probability(n_wifi) > jax.random.uniform(key)

        n_successful = n_successful * (1 - collision)
        n_failed = n_all - n_successful

        cw = jnp.where(n_successful > 0, MIN_CW_EXP, state.cw + 1)
        cw = jnp.where(cw <= MAX_CW_EXP, cw, MAX_CW_EXP)

        state = RASimState(
            time=state.time,
            snr=state.snr,
            ptr=state.ptr + 1,
            cw=cw
        )

        terminated = state.ptr == len(state.time)
        reward = jnp.where(n_all > 0, phy_rates[action] * n_successful / n_all, 0.0)

        return state, _get_env_state(state, action, n_successful, n_failed), reward, terminated

    return Env(
        init=init,
        step=step
    )


class RASimEnv(gym.Env):
    """
    Ra-sim simulator for the IEEE 802.11ax networks. Calculates if packet has been transmitted
    successfully based on the approximated success probability based on the current distance
    (transformed to the signal-to-noise ratio (SNR) according to the log-distance channel model)
    and approximated collision probability (derived empirically). The environment simulates
    the IEEE 802.11ax networks with the following settings: a guard interval is equal to 3200 ns,
    channel width is 20 MHz, 1 spatial stream is used, and the packets are treated as indivisible
    with Aggregated MAC Protocol Data Unit (A-MPDU) batches of frames of size 1500 B.
    """

    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Dict({
            'time': gym.spaces.Box(0.0, jnp.inf, (1,)),
            'n_successful': gym.spaces.Box(0, jnp.inf, (1,), jnp.int32),
            'n_failed': gym.spaces.Box(0, jnp.inf, (1,), jnp.int32),
            'n_wifi': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'power': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'cw': gym.spaces.Discrete(32767),
            'mcs': gym.spaces.Discrete(12)
        })

        self.options = {
            'initial_position': 0.0,
            'n_wifi': 1,
            'simulation_time': 25.0,
            'velocity': 2.0
        }

    def reset(
            self,
            seed: int = None,
            options: Dict = None
    ) -> Tuple[gym.spaces.Dict, Dict]:
        """
        Resets the environment to the initial state.

        Parameters
        ----------
        seed : int
            Integer used as the random key.
        options : dict
            Dictionary containing simulation parameters, i.e. ``initial_position``, ``n_wifi``,
            ``simulation_time``, ``velocity``.

        Returns
        -------
        tuple[dict, dict]
            Initial environment state.
        """

        seed = seed if seed else random.randint(0, sys.maxsize)
        super().reset(seed=seed)
        self.key = jax.random.PRNGKey(seed)

        options = options if options else {}
        self.options.update(options)

        self.sim = ra_sim(
            self.options['simulation_time'],
            self.options['velocity'],
            self.options['initial_position'],
            self.options['n_wifi'],
            int(self.options['simulation_time'] * FRAMES_PER_SECOND)
        )
        self.state, env_state = self.sim.init()

        return env_state, {}

    def step(self, action: int) -> Tuple[gym.spaces.Dict, float, bool, bool, Dict]:
        """
        Performs one step of the environment and returns new environment state.

        Parameters
        ----------
        action : int
            Action to perform in the environment.

        Returns
        -------
        tuple[dict, float, bool, bool, dict]
            Environment state after performing an action (i.e., observations, reward, and info about termination).
        """

        step_key, self.key = jax.random.split(self.key)
        self.state, *env_state = self.sim.step(self.state, action, step_key)

        return *env_state, False, {}
