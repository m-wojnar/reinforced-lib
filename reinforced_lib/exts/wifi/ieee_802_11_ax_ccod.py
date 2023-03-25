import gymnasium as gym
import numpy as np

from reinforced_lib.exts import BaseExt, observation, parameter


class IEEE_802_11_ax_CCOD(BaseExt):
    """
    The IEEE 802.11ax [2]_ extension. Provides the data rates (in Mb/s) for consecutive modulation and coding
    schemes (MCS), a reward calculated as the approximate throughput, environment state as a vector, and
    default parameters for MAB and DRL agents. This extension is adapted to the IEEE 802.11ax network with the
    following settings: the guard interval is equal to 3200 ns, channel width is 20 MHz, 1 spatial stream is used.

    References
    ----------
    .. [2] "IEEE Standard for Information Technology--Telecommunications and Information Exchange between Systems
       Local and Metropolitan Area Networks--Specific Requirements Part 11: Wireless LAN Medium Access Control
       (MAC) and Physical Layer (PHY) Specifications Amendment 1: Enhancements for High-Efficiency WLAN,"
       in IEEE Std 802.11ax-2021 (Amendment to IEEE Std 802.11-2020) , vol., no., pp.1-767,
       19 May 2021, doi: 10.1109/IEEESTD.2021.9442429.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_time = 0.0

    observation_space = gym.spaces.Dict({
        'history_sie': gym.spaces.Box(0.0, np.inf, (1,)),
        'history': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'reward': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'sim_time': gym.spaces.Box(1, np.inf, (1,), np.int32),
        'power': gym.spaces.Box(-np.inf, np.inf, (1,)),
        'cw': gym.spaces.Discrete(32767)
    })

    @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def reward(self, action: int, n_successful: int, n_failed: int, *args, **kwargs) -> float:
        if n_successful + n_failed > 0:
            return self._wifi_modes_rates[action] * n_successful / (n_successful + n_failed)
        else:
            return 0.0

    @observation(observation_type=gym.spaces.Box(0.0, np.inf, (1,)))
    def delta_time(self, time: float, *args, **kwargs) -> float:
        delta_time = time - self.last_time
        self.last_time = time
        return delta_time

    @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (6,)))
    def env_state(
            self,
            time: float,
            n_successful: int,
            n_failed: int,
            n_wifi: int,
            power: float,
            cw: int,
            *args,
            **kwargs
    ) -> np.ndarray:
        return np.array([self.delta_time(time), n_successful, n_failed, n_wifi, power, cw], dtype=np.float32)

    @observation(observation_type=gym.spaces.MultiBinary(1))
    def terminal(self, *args, **kwargs) -> bool:
        return False

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def n_mcs(self) -> int:
        return len(self._wifi_modes_rates)

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def n_arms(self) -> int:
        return self.n_mcs()

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def default_power(self) -> float:
        return 16.0206

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def min_reward(self) -> float:
        return 0

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def max_reward(self) -> int:
        return self._wifi_modes_rates.max()

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, np.inf, (1,), np.int32)))
    def obs_space_shape(self) -> tuple:
        return tuple((6,))

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, np.inf, (1,), np.int32)))
    def act_space_shape(self) -> tuple:
        return tuple((1,))

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def act_space_size(self) -> int:
        return 12
