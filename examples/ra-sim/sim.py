from typing import Dict, Tuple

import gym
import numpy as np
from gym import spaces

from reinforced_lib.exts import IEEE_802_11_ax


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
AMPDU_SIZES = np.array([3, 6, 9, 12, 18, 25, 28, 31, 37, 41, 41, 41])

# default values used by the ns-3 simulator
# https://www.nsnam.org/docs/models/html/wifi-testing.html#packet-error-rate-performance
DEFAULT_NOISE = -93.97
DEFAULT_TX_POWER = 16.0206

# LogDistance channel model
# https://www.nsnam.org/docs/models/html/propagation.html#logdistancepropagationlossmodel
REFERENCE_SNR = DEFAULT_TX_POWER - DEFAULT_NOISE
REFERENCE_LOSS = 46.6777
EXPONENT = 3.0

def distance_to_snr(distance: np.ndarray) -> np.ndarray:
    return REFERENCE_SNR - (REFERENCE_LOSS + 10 * EXPONENT * np.log10(distance))


class RASimEnv(gym.Env):
    """
    Simple Rate Adaptation Simulator for IEEE 802.11ax networks. Calculates if packet has been transmitted
    successfully based on approximated success probability for a given distance and SNR (according to the
    LogDistance channel model) and approximated collision probability (calculated experimentally).
    Environment simulates Wi-Fi networks with 20 MHz width channel, guard interval set to 3200 ns,
    1 spatial stream, and the packet is treated as indivisible.
    """

    def __init__(self) -> None:
        self.action_space = spaces.Discrete(12)
        self.observation_space = gym.spaces.Dict({
            'time': gym.spaces.Box(0.0, np.inf, (1,)),
            'n_successful': gym.spaces.Box(0, np.inf, (1,), np.int32),
            'n_failed': gym.spaces.Box(0, np.inf, (1,), np.int32),
            'n_wifi': gym.spaces.Box(1, np.inf, (1,), np.int32),
            'power': gym.spaces.Box(-np.inf, np.inf, (1,)),
            'cw': gym.spaces.Discrete(32767),
            'mcs': gym.spaces.Discrete(12)
        })

        self.options = {
            'initial_position': 0.0,
            'n_wifi': 1,
            'simulation_time': 25.0,
            'velocity': 2.0
        }

        self.ieee_802_11_ax = IEEE_802_11_ax()

    def reset(
            self,
            seed: int = None,
            options: Dict = None
    ) -> Tuple[gym.spaces.Dict, Dict]:
        """
        Sets the environment to the initial state.

        Parameters
        ----------
        seed : int
            An integer used as the random key.
        options : dict
            Dictionary containing simulation parameters, i.e. `initial_position`, `n_wifi`, `simulation_time`, `velocity`.

        Returns
        -------
        state : tuple[dict, dict]
            Initial environment state.
        """

        super().reset(seed=seed)

        options = options if options else {}
        self.options.update(options)

        total_time = self.options['simulation_time']
        total_distance = total_time * self.options['velocity']
        total_frames = int(total_time * FRAMES_PER_SECOND)

        distance = np.abs(np.linspace(0.0, total_distance, total_frames) + self.options['initial_position'])
        self.snr = distance_to_snr(distance)
        self.time = np.linspace(0.0, total_time, total_frames)
        self.ptr = 0

        self.cw = MIN_CW_EXP
        self.last_tx_successful = 1

        state = {
            'time': 0.0,
            'n_successful': 0,
            'n_failed': 0,
            'n_wifi': self.options['n_wifi'],
            'power': DEFAULT_TX_POWER,
            'cw': 2 ** self.cw - 1,
            'mcs': 0
        }

        return state, {}

    def step(self, action: int) -> Tuple[gym.spaces.Dict, float, bool, bool, Dict]:
        """
        Performs one step in the environment and returns new environment state.

        Parameters
        ----------
        action : int
            Action to perform in the environment.

        Returns
        -------
        out : tuple[dict, float, bool, bool, dict]
            Environment state after performing a step, reward, and info about termination.
        """

        n_all = AMPDU_SIZES[action]
        n_successful = int(n_all * self.ieee_802_11_ax.success_probability(self.snr[self.ptr])[action])
        collision = self.ieee_802_11_ax.collision_probability(self.options['n_wifi']) > np.random.random()

        n_successful = n_successful * (1 - collision)
        n_failed = n_all - n_successful

        reward = self.ieee_802_11_ax.reward(action, n_successful, n_failed)

        if n_successful > 0:
            self.cw = MIN_CW_EXP
        else:
            self.cw = min(self.cw + 1, MAX_CW_EXP)

        state = {
            'time': self.time[self.ptr],
            'n_successful': n_successful,
            'n_failed': n_failed,
            'n_wifi': self.options['n_wifi'],
            'power': DEFAULT_TX_POWER,
            'cw': 2 ** self.cw - 1,
            'mcs': action
        }

        self.ptr += 1
        done = self.ptr == len(self.time)

        return state, reward, done, False, {}
