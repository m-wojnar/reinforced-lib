import gymnasium as gym
import jax.numpy as jnp

from reinforced_lib.exts import BaseExt, observation, parameter


class GymnasiumVectorized(BaseExt):
    """
    Vectorized Gymnasium [1]_ extension. Simplifies interaction of RL agents with the vectorized Gymnasium environments
    by providing the environment state, reward, terminal flag, and shapes of the observation and action spaces.

    Parameters
    ----------
    env_id : str
        Name of the Gymnasium environment.
    num_envs : int
        Number of parallel environments.
    """

    def __init__(self, env_id: str, num_envs: int) -> None:
        self.env = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(num_envs)])
        super().__init__()

    observation_space = gym.spaces.Dict({})

    @observation()
    def env_states(self, env_states, rewards, terminals, truncated, infos, *args, **kwargs) -> any:
        return env_states

    @observation()
    def rewards(self, env_states, rewards, terminals, truncated, infos, *args, **kwargs) -> float:
        return rewards

    @observation()
    def terminals(self, env_states, rewards, terminals, truncated, infos, *args, **kwargs) -> bool:
        return terminals | truncated

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), int)))
    def obs_space_shape(self) -> tuple:
        return self.env.single_observation_space.shape

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), int)))
    def act_space_shape(self) -> tuple:
        return self.env.single_action_space.shape

    @parameter(parameter_type=gym.spaces.Box(1, jnp.inf, (1,), int))
    def act_space_size(self) -> int:
        if isinstance(self.env.single_action_space, gym.spaces.Discrete):
            return self.env.single_action_space.n

        raise AttributeError()

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)))
    def min_action(self) -> tuple:
        return self.env.single_action_space.low

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)))
    def max_action(self) -> tuple:
        return self.env.single_action_space.high
