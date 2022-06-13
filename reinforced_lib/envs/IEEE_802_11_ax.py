from typing import Any

import gym.spaces
import numpy as np

from envs.utils import observation
from reinforced_lib.envs.base_env import BaseEnv


class IEEE_802_11_ax(BaseEnv):
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

    observation_space = gym.spaces.Dict({
        'time': gym.spaces.Box(0.0, np.inf, (1,)),
        'n_successful': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'n_failed': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'n_wifi': gym.spaces.Box(1, np.inf, (1,), np.int32),
        'power': gym.spaces.Box(-np.inf, np.inf, (1,)),
        'cw': gym.spaces.Discrete(32767)
    })

    action_space = gym.spaces.Discrete(len(_wifi_modes_rates))

    @observation(parameter_type=gym.spaces.Box(0.0, 1.0))
    def collision_probability(self, n_wifi: int, *args, **kwargs):
        return 0.154887 * np.log(1.03102 * n_wifi)

    def reset(self) -> None:
        pass

    def act(self, *args, **kwargs) -> Any:
        pass
