import gym.spaces
import numpy as np

from reinforced_lib.envs.base_env import BaseEnv
from reinforced_lib.envs.utils import observation


class IEEE_802_11_ax(BaseEnv):
    def __init__(self, agent_update_space: gym.spaces.Space, agent_sample_space: gym.spaces.Space) -> None:
        """
        IEEE 802.11ax [1] environment. Provides data rates (in Mb/s) for consecutive MCS (modulation and coding
        scheme) modes, minimal SNR (signal-to-noise ratio) for each mode and approximated collision probability
        for given number of transmitting stations.

        Parameters
        ----------
        agent_update_space : gym.spaces.Space
            Parameters required by the agents 'update' function in OpenAI Gym format.
        agent_sample_space : gym.spaces.Space
            Parameters required by the agents 'sample' function in OpenAI Gym format.

        References
        ----------
        .. [1] "IEEE Standard for Information Technology--Telecommunications and Information Exchange between Systems
            Local and Metropolitan Area Networks--Specific Requirements Part 11: Wireless LAN Medium Access Control
            (MAC) and Physical Layer (PHY) Specifications Amendment 1: Enhancements for High-Efficiency WLAN,"
            in IEEE Std 802.11ax-2021 (Amendment to IEEE Std 802.11-2020) , vol., no., pp.1-767,
            19 May 2021, doi: 10.1109/IEEESTD.2021.9442429.

        """

        super().__init__(agent_update_space, agent_sample_space)

    observation_space = gym.spaces.Dict({
        'time': gym.spaces.Box(0.0, np.inf, (1,)),
        'n_successful': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'n_failed': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'n_wifi': gym.spaces.Box(1, np.inf, (1,), np.int32),
        'power': gym.spaces.Box(-np.inf, np.inf, (1,)),
        'cw': gym.spaces.Discrete(32767)
    })

    _wifi_modes_rates = np.array([
        7.3,
        14.6,
        21.9,
        29.3,
        43.9,
        58.5,
        65.8,
        73.1,
        87.8,
        97.5,
        109.7,
        121.9
    ])

    _wifi_modes_snrs = np.array([
        0.48949,
        3.42567,
        6.54673,
        9.48308,
        13.0948,
        16.9275,
        18.9046,
        20.6119,
        24.0752,
        25.8097,
        31.7291,
        33.6907,
    ])

    @observation(observation_type=gym.spaces.Box(0.0, np.inf, (len(_wifi_modes_rates),)))
    def rates(self, *args, **kwargs) -> np.ndarray:
        return self._wifi_modes_rates

    @observation(observation_type=gym.spaces.Box(0.0, np.inf, (len(_wifi_modes_snrs),)))
    def min_snr(self, *args, **kwargs) -> np.ndarray:
        return self._wifi_modes_snrs

    @observation(observation_type=gym.spaces.Box(0.0, np.inf, (len(_wifi_modes_rates),)))
    def context(self, *args, **kwargs) -> np.ndarray:
        return self._wifi_modes_rates

    @observation(observation_type=gym.spaces.Box(0.0, 1.0, (1,)))
    def collision_probability(self, n_wifi: int, *args, **kwargs) -> float:
        return 0.154887 * np.log(1.03102 * n_wifi)
