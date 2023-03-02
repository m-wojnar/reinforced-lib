from typing import Dict, Tuple
import gymnasium as gym
import numpy as np


gym.envs.registration.register(
    id='RecommenderSystemEnv-v1',
    entry_point='examples.recommender_system.env:RecommenderSystemEnv'
)


class RecommenderSystemEnv(gym.Env):
    """
    Simple Multi Armed Bandit environment to test solutions of recommending some specific goods 
    for the user. The environment consists of a finite number of goods which are mapped by the users
    preference function. In each step a representative of goods is presented for the user, and he can
    either like or dislike it. The reward is 1 or 0 accordingly.
    """

    def __init__(self, preferences: Dict) -> None:

        self.action_space = gym.spaces.Discrete(len(preferences))
        self.observation_space = gym.spaces.Space()
        self._preferences = list(preferences.values())  # preferences as probability of enjoying the content
    
    def reset(
            self,
            seed: int = None,
            options: Dict = None
    ) -> Tuple[gym.spaces.Space, Dict]:
        """
        Resets the environment to the initial state.

        Parameters
        ----------
        seed : int
            An integer used as the random key.
        options : dict
            Dictionary containing environment options. Ignored by default, method iherited form
            `gem.Env`

        Returns
        -------
        state : tuple[Space, dict]
            Initial environment state.
        """

        seed = seed if seed else np.random.randint(1000)
        super().reset(seed=seed)
        np.random.seed(seed)

        return None, {}
    
    def step(self, action: int) -> Tuple[gym.spaces.Dict, int, bool, bool, Dict]:
        """
        Reacts to the agent action (presentation of some good) by returning the reward
        from the user, which can be interpreted as either he or she liked or disliked the acion (good).

        Parameters
        ----------
        action : int
            Action to perform in the environment, can be interpreted as an id ofsome good.

        Returns
        -------
        out : tuple[None, int, bool, bool, dict]
            Environment state as None, the reward and auxiliary information about the state. Here
            in MAB problem only reward is changing, one can ignore the rest.
        """

        reward = int(np.random.rand() < self._preferences[action])
        
        return None, reward, False, False, {}
