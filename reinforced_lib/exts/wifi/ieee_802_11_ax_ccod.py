from chex import Array

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

    def __init__(self, history_length: int) -> None:
        super().__init__()
        self.history_length = history_length
        self.last_time = 0.0

    max_history_length = 512
    no_actions = 6

    observation_space = gym.spaces.Dict({
        'history_sie': gym.spaces.Box(0.0, np.inf, (1,)),
        'history': gym.spaces.Box(0, 1, (max_history_length,), np.int32),
        'reward': gym.spaces.Box(-np.inf, np.inf, (1,)),
        'sim_time': gym.spaces.Box(0, np.inf, (1,)),
        'current_thr': gym.spaces.Box(0, np.inf, (1,)),
        'n_wifi': gym.spaces.Box(0, np.inf, (1,), np.int32)
    })

    @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def reward(self, reward: float, *args, **kwargs) -> float:
        return reward

    @observation(observation_type=gym.spaces.Box(0.0, np.inf, (1,)))
    def delta_time(self, time: float, *args, **kwargs) -> float:
        delta_time = time - self.last_time
        self.last_time = time
        return delta_time

    # TODO Jak przekazać historię, która ma zmienny rozmiar w zaleności od 'history_length'?
    @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (max_history_length,)))
    def env_state(
            self,
            history_sie: int,
            history: Array,
            *args,
            **kwargs
    ) -> np.ndarray:
        return np.array(history[:history_sie], dtype=np.float32)

    @observation(observation_type=gym.spaces.MultiBinary(1))
    def terminal(self, *args, **kwargs) -> bool:
        return False

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def min_reward(self) -> float:
        return 0.

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def max_reward(self) -> float:
        return 1.

    # TODO jak się uda do env_state wrzucić history_length, to tu te trzeba
    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(0, np.inf, (1,), np.int32)))
    def obs_space_shape(self) -> tuple:
        return tuple((self.max_history_length,))

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, np.inf, (1,), np.int32)))
    def act_space_shape(self) -> tuple:
        return tuple((1,))

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def act_space_size(self) -> int:
        return 6
