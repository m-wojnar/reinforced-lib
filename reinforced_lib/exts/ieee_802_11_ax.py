import gym.spaces
import numpy as np

from reinforced_lib.exts import BaseExt, observation, parameter


class IEEE_802_11_ax(BaseExt):
    """
    The IEEE 802.11ax [1]_ extension. Provides the data rates (in Mb/s) for consecutive modulation and coding
    schemes (MCS), a reward calculated as an approximated throughput, and the default transmission power.
    This extension is adapted to the IEEE 802.11ax network with the following settings: a guard interval is
    equal to 3200 ns, channel width is 20 MHz, 1 spatial stream is used.

    References
    ----------
    .. [1] "IEEE Standard for Information Technology--Telecommunications and Information Exchange between Systems
       Local and Metropolitan Area Networks--Specific Requirements Part 11: Wireless LAN Medium Access Control
       (MAC) and Physical Layer (PHY) Specifications Amendment 1: Enhancements for High-Efficiency WLAN,"
       in IEEE Std 802.11ax-2021 (Amendment to IEEE Std 802.11-2020) , vol., no., pp.1-767,
       19 May 2021, doi: 10.1109/IEEESTD.2021.9442429.

    """

    observation_space = gym.spaces.Dict({
        'time': gym.spaces.Box(0.0, np.inf, (1,)),
        'n_successful': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'n_failed': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'n_wifi': gym.spaces.Box(1, np.inf, (1,), np.int32),
        'power': gym.spaces.Box(-np.inf, np.inf, (1,)),
        'cw': gym.spaces.Discrete(32767),
        'mcs': gym.spaces.Discrete(12)
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
    ], dtype=np.float32)

    @observation(observation_type=gym.spaces.Box(0.0, np.inf, (len(_wifi_modes_rates),)))
    def rates(self, *args, **kwargs) -> np.ndarray:
        return self._wifi_modes_rates

    @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (len(_wifi_modes_rates),)))
    def context(self, *args, **kwargs) -> np.ndarray:
        return self.rates()

    @observation(observation_type=gym.spaces.Discrete(len(_wifi_modes_rates)))
    def action(self, mcs: int, *args, **kwargs) -> int:
        return mcs

    @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def reward(self, mcs: int, n_successful: int, n_failed: int, *args, **kwargs) -> float:
        if n_successful + n_failed > 0:
            return self._wifi_modes_rates[mcs] * n_successful / (n_successful + n_failed)
        else:
            return 0.0

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def n_mcs(self) -> int:
        return len(self._wifi_modes_rates)

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def n_arms(self) -> int:
        return self.n_mcs()

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def default_power(self) -> float:
        return 16.0206
