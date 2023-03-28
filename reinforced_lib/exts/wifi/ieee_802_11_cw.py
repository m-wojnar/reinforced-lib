from chex import Array

import gymnasium as gym
import numpy as np

from reinforced_lib.exts import BaseExt, observation, parameter


class IEEE_802_11_CW(BaseExt):
    history_length = 20

    observation_space = gym.spaces.Dict({
        'history': gym.spaces.Box(0, 1, (history_length,), np.int32),
        'reward': gym.spaces.Box(-np.inf, np.inf, (1,))
    })

    @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (history_length, 1), np.float32))
    def env_state(self, history: Array, *args, **kwargs) -> np.ndarray:
        return np.array(history, dtype=np.float32)[..., np.newaxis]

    @observation(observation_type=gym.spaces.MultiBinary(1))
    def terminal(self, *args, **kwargs) -> bool:
        return False

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def min_reward(self) -> float:
        return 0.

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def max_reward(self) -> float:
        return 1.

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(0, np.inf, (1,), np.int32)))
    def obs_space_shape(self) -> tuple:
        return self.history_length, 1

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, np.inf, (1,), np.int32)))
    def act_space_shape(self) -> tuple:
        return tuple((1,))

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def act_space_size(self) -> int:
        return 6
