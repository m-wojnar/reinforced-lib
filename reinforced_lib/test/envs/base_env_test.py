import gym.spaces
import numpy as np
from numpy import ndarray

from reinforced_lib.envs.base_env import BaseEnv
from reinforced_lib.envs.utils import observation


class DummyAgent:
    update_observation_space = gym.spaces.Dict({
        'time': gym.spaces.Box(0.0, 100.0, (1,)),
        'n': gym.spaces.Discrete(10),
        'params': gym.spaces.Tuple([
            gym.spaces.Discrete(20),
            gym.spaces.Discrete(30),
            gym.spaces.Dict({
                'test': gym.spaces.Discrete(40)
            })
        ])
    })

    sample_observation_space = gym.spaces.Dict({
        'matrix': gym.spaces.Box(0.0, 1.0, (10, 2))
    })

    action_space = gym.spaces.Discrete(10)


class DummyEnv(BaseEnv):
    def __init__(self, agent_update_space: gym.spaces.Space, agent_sample_space: gym.spaces.Space) -> None:
        super().__init__(agent_update_space, agent_sample_space)

    observation_space = gym.spaces.Dict({
        'n': gym.spaces.Discrete(10),
        'params': gym.spaces.Tuple([
            gym.spaces.Discrete(20),
            gym.spaces.Discrete(30),
            gym.spaces.Dict({
                'test': gym.spaces.Discrete(40)
            })
        ]),
        'not_used': gym.spaces.MultiBinary((12, 15))
    })

    @observation(parameter_type=gym.spaces.Box(0.0, 1.0, (10, 2)))
    def matrix(self, arg: float, *args, **kwargs) -> ndarray:
        assert 0.0 <= arg <= 1.0
        return np.full((10, 2), arg)

    @observation('time')
    def dummy_function(self, *args, **kwargs) -> float:
        return 0.1234


if __name__ == '__main__':
    agent = DummyAgent()
    env = DummyEnv(agent.update_observation_space, agent.sample_observation_space)

    print(env._update_space_transform(0.5, n=5, params=(10, 15, {'test': 20})))
    print(env._sample_space_transform(0.5, n=5, params=(10, 15, {'test': 20})))
