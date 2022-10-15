from typing import Dict
from chex import Scalar, Array
import numpy as np
import gym.spaces

from reinforced_lib.exts import BaseExt, observation, parameter


class RecommenderSystemExt(BaseExt):

    observation_space = gym.spaces.Space()

    def __init__(self, preferences: Dict, e: Scalar = 0.1) -> None:
        super().__init__()
        self._preferences = preferences
        self._content_tags = list(preferences.keys())
        self._e = e
    
    @observation()
    def action(self, action, *args, **kwargs):
        return action

    @observation()
    def context(self, *args, **kwargs):
        return np.ones(self.n_arms())
    
    @observation()
    def reward(self, action, *args, **kwargs):
        return int(np.random.rand() < self._preferences[self._content_tags[action]])
    
    @parameter(parameter_type=gym.spaces.Box(0.0, 1.0, (1,), np.float32))
    def e(self):
        return self._e

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def n_arms(self):
        return len(self._preferences)
