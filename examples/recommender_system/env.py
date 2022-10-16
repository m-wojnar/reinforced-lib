from typing import Dict, Tuple
import gym
import gym.spaces
import numpy as np


gym.envs.registration.register(
    id='RecommenderSystemEnv-v1',
    entry_point='examples.recommender_system.env:RecommenderSystemEnv'
)


class RecommenderSystemEnv(gym.Env):

    def __init__(self, preferences: Dict) -> None:
        self.action_space = gym.spaces.Discrete(len(preferences))
        self.observation_space = gym.spaces.Space()

        self._preferences = list(preferences.values())  # preferences as probability of enjoying the content
    
    def reset(
            self,
            seed: int = None,
            options: Dict = None
    ) -> Tuple[gym.spaces.Space, Dict]:

        seed = seed if seed else np.random.randint(1000)
        super().reset(seed=seed)
        np.random.seed(seed)

        return None, {}
    
    def step(self, action: int) -> Tuple[gym.spaces.Dict, float, bool, bool, Dict]:
        reward = int(np.random.rand() < self._preferences[action])
        return None, reward, False, False, {}
