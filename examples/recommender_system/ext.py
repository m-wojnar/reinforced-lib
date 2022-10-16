import gym.spaces
import numpy as np

from reinforced_lib.exts import BaseExt, observation, parameter


class RecommenderSystemExt(BaseExt):
    preferences = {
        'astronomy': 0.71,      # 0
        'nature': 0.6,          # 1
        'cooking': 0.6,         # 2
        'games': 0.2,           # 3
        'music': 0.92,          # 4 Highest expected reward
        'sports': 0.4,          # 5
        'technology': 0.67      # 6
    }
    
    observation_space = gym.spaces.Dict({
        'action': gym.spaces.Discrete(len(preferences)),
        'reward': gym.spaces.Box(-np.inf, np.inf, (1,)),
        'time': gym.spaces.Box(0.0, np.inf, (1,))
    })

    def __init__(self) -> None:
        super().__init__()
        self._preferences = list(self.preferences.values())
        self._context = np.ones(len(self.preferences))

    @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (len(preferences),), np.float32))
    def context(self, *args, **kwargs):
        return self._context

    @observation(observation_type=gym.spaces.Box(0, np.inf, (1,), np.int32))
    def n_successful(self, reward: float, *args, **kwargs):
        return reward

    @observation(observation_type=gym.spaces.Box(0, np.inf, (1,), np.int32))
    def n_failed(self, reward: float, *args, **kwargs):
        return 1 - reward

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def n_arms(self):
        return len(self._preferences)
