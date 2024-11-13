import gymnasium as gym
import jax.numpy as jnp

from reinforced_lib.exts import BaseExt, observation, parameter


class RecommenderSystemExt(BaseExt):
    """
    Basic recommender system extension where we can present the user a content in one of 7 categories
    (astronomy, nature, cooking, games, music, sports or technology) and we must infer which one of those is
    he or she most interested in. The true interest of the user in 0-1 scale is stored in the ``preferences``
    dictionary attribute. Extension provides parameters and observations for MAB agents.
    """

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
        'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float),
        'time': gym.spaces.Box(0.0, jnp.inf, (1,), float),
    })

    def __init__(self) -> None:
        super().__init__()
        self._preferences = list(self.preferences.values())
        self._context = jnp.ones(len(self.preferences))

    @observation(observation_type=gym.spaces.Box(-jnp.inf, jnp.inf, (len(preferences),), float))
    def context(self, *args, **kwargs):
        return self._context

    @observation(observation_type=gym.spaces.Box(0, jnp.inf, (1,), int))
    def n_successful(self, reward: float, *args, **kwargs):
        return reward

    @observation(observation_type=gym.spaces.Box(0, jnp.inf, (1,), int))
    def n_failed(self, reward: float, *args, **kwargs):
        return 1 - reward

    @parameter(parameter_type=gym.spaces.Box(1, jnp.inf, (1,), int))
    def n_arms(self):
        return len(self._preferences)
