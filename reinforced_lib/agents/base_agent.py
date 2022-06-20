from abc import ABC, abstractmethod
from typing import Any, Tuple

import chex
import gym.spaces

from reinforced_lib.agents.agent_state import AgentState


class BaseAgent(ABC):
    """
    Container for functions of the agent, observation spaces, and action space.
    """

    @staticmethod
    @abstractmethod
    def init(*args, **kwargs) -> AgentState:
        """
        Creates and initializes instance of the agent.
        """

        pass

    @staticmethod
    @abstractmethod
    def update(state: AgentState, key: chex.PRNGKey, *args, **kwargs) -> AgentState:
        """
        Updates the state of the agent after performing some action and receiving a reward.
        """

        pass

    @staticmethod
    @abstractmethod
    def sample(state: AgentState, key: chex.PRNGKey, *args, **kwargs) -> Tuple[AgentState, Any]:
        """
        Selects next action based on current agent state.
        """

        pass

    @staticmethod
    @abstractmethod
    def init_observation_space() -> gym.spaces.Dict:
        """
        Parameters required by the 'init' method in OpenAI Gym format.
        Type of returned value is required to be gym.spaces.Dict.
        """

        pass

    @property
    @abstractmethod
    def update_observation_space(self) -> gym.spaces.Space:
        """
        Parameters required by the 'update' method in OpenAI Gym format.
        """

        pass

    @property
    @abstractmethod
    def sample_observation_space(self) -> gym.spaces.Space:
        """
        Parameters required by the 'sample' method in OpenAI Gym format.
        """

        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        """
        Action returned by the agent in OpenAI Gym format.
        """

        pass
