from typing import Tuple, Dict, Any

import gym.spaces
import numpy as np
from numpy import ndarray

from reinforced_lib.envs.utils import observation
from reinforced_lib.agents.agent_state import AgentState
from reinforced_lib.agents.base_agent import BaseAgent
from reinforced_lib.envs.base_env import BaseEnv
from reinforced_lib.envs.env_state import EnvState


class DummyAgent:
    def __init__(self):
        self.update_observation_space = gym.spaces.Discrete(n=20)
        self.sample_observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(10, 2))


class DummyEnv(BaseEnv):
    observation_space = gym.spaces.Discrete(n=20)
    action_space = None

    def __init__(self, agent: BaseAgent, agent_state: AgentState) -> None:
        super().__init__(agent, agent_state)

    @observation(parameter_type=gym.spaces.Box(low=0.0, high=1.0, shape=(10, 2)))
    def dummy_function(self, arg: float) -> ndarray:
        assert 0.0 <= arg <= 1.0
        return np.full((10, 2), arg)

    def reset(self) -> EnvState:
        pass

    def act(self, *args: Tuple, **kwargs: Dict) -> Any:
        pass


if __name__ == '__main__':
    agent = DummyAgent()
    env = DummyEnv(agent, None)

    print(env.update_space_transform(10))
    print(env.sample_space_transform(0.5))
