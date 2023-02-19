from typing import Any

import gymnasium as gym
import numpy as np

from reinforced_lib.exts import BaseExt, observation, parameter


class Gym(BaseExt):
    """
    Gym extension. Simplifies usage of deep RL agents by providing environment state, reward, terminal flag,
    and shapes of the observation and action spaces.

    Parameters
    ----------
    env_id : str
        Name of the Gym environment.
    """

    def __init__(self, env_id: str) -> None:
        self.env = gym.make(env_id)
        super().__init__()

    observation_space = gym.spaces.Dict({})

    @observation()
    def env_state(self, env_state, reward, terminal, truncated, info, *args, **kwargs) -> Any:
        return env_state

    @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def reward(self, env_state, reward, terminal, truncated, info, *args, **kwargs) -> float:
        return reward

    @observation(observation_type=gym.spaces.MultiBinary(1))
    def terminal(self, env_state, reward, terminal, truncated, info, *args, **kwargs) -> bool:
        return terminal or truncated

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, np.inf, (1,), np.int32)))
    def obs_space_shape(self) -> tuple:
        return self.env.observation_space.shape

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, np.inf, (1,), np.int32)))
    def act_space_shape(self) -> tuple:
        return self.env.action_space.shape

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def act_space_size(self) -> int:
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return self.env.action_space.n

        raise AttributeError()
