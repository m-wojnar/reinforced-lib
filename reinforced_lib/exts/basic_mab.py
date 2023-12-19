import gymnasium as gym
import jax.numpy as jnp

from reinforced_lib.exts import BaseExt, observation, parameter


class BasicMab(BaseExt):
    """
    Basic multi-armed bandit (MAB) extension for Reinforced-lib. This extension can be used with MAB algorithms
    which do not require any additional information about the environment apart from the number of arms.
    """

    def __init__(self, n_arms: int) -> None:
        super().__init__()
        self.n = n_arms

    observation_space = gym.spaces.Dict({})

    @parameter(parameter_type=gym.spaces.Box(1, jnp.inf, (1,), int))
    def n_arms(self) -> int:
        return self.n

    @observation(observation_type=gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float))
    def reward(self, reward, *args, **kwargs) -> float:
        return reward
