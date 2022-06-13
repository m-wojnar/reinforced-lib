from typing import Tuple, Dict, Any

import gym.spaces
import numpy as np
from numpy import ndarray

from reinforced_lib.agents.agent_state import AgentState
from reinforced_lib.agents.base_agent import BaseAgent
from reinforced_lib.envs.base_env import BaseEnv
from reinforced_lib.envs.env_state import EnvState
from reinforced_lib.envs.utils import observation


class DummyAgent:
    update_observation_space = gym.spaces.Dict({
        'time': gym.spaces.Box(0.0, 100.0, (1,)),
        'n': gym.spaces.Discrete(10),
        'params': gym.spaces.Tuple([
            gym.spaces.Discrete(20),
            gym.spaces.Discrete(30)
        ])
    })

    sample_observation_space = gym.spaces.Dict({
        'matrix': gym.spaces.Box(low=0.0, high=1.0, shape=(10, 2))
    })

    action_space = gym.spaces.Discrete(10)


class DummyEnv(BaseEnv):
    observation_space = gym.spaces.Dict({
        'n': gym.spaces.Discrete(10),
        'params': gym.spaces.Tuple([
            gym.spaces.Discrete(20),
            gym.spaces.Discrete(30)
        ]),
        'not_used': gym.spaces.MultiBinary((12, 15))
    })

    action_space = None

    def __init__(self, agent: BaseAgent, agent_state: AgentState) -> None:
        super().__init__(agent, agent_state)
        self.action_space = agent.action_space

    @observation(parameter_type=gym.spaces.Box(low=0.0, high=1.0, shape=(10, 2)))
    def matrix(self, arg: float) -> ndarray:
        assert 0.0 <= arg <= 1.0
        return np.full((10, 2), arg)

    @observation('time')
    def dummy_function(self) -> float:
        return 0.1234

    def reset(self) -> EnvState:
        pass

    def act(self, *args: Tuple, **kwargs: Dict) -> Any:
        pass


if __name__ == '__main__':
    agent = DummyAgent()
    env = DummyEnv(agent, None)

    obs = {
        'n': 5,
        'params': [10, 15]
    }

    print(env.update_space(obs))
    print(env.sample_space(obs))
