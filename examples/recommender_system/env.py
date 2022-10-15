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
        super().__init__()
        self.content_tags = list(preferences.keys())
        self.preferences = preferences  # preferences as probability of enjoying the content
        self.action_space = gym.spaces.Discrete(len(self.content_tags))
        self.observation_space = gym.spaces.Space()
    
    def reset(self, seed: int = None) -> Tuple[gym.spaces.Space, Dict]:

        self.seed = seed if seed else np.random.randint(1000)
        super().reset(seed=self.seed)
        np.random.seed(self.seed)

        return None, {}
    
    def step(self, action: int) -> Tuple[gym.spaces.Dict, float, bool, bool, Dict]:

        reward = int(np.random.rand() < self.preferences[self.content_tags[action]])

        return None, reward, False, False, {}

