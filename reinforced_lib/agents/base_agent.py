from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
from chex import dataclass, PRNGKey


@dataclass
class AgentState:
    """
    Base class for agent state containers.
    """


class BaseAgent(ABC):
    """
    Base interface of agents.
    """

    @staticmethod
    @abstractmethod
    def init(key: PRNGKey, *args, **kwargs) -> AgentState:
        """
        Creates and initializes instance of the agent.
        """

        pass

    @staticmethod
    @abstractmethod
    def update(state: AgentState, key: PRNGKey, *args, **kwargs) -> AgentState:
        """
        Updates the state of the agent after performing some action and receiving a reward.
        """

        pass

    @staticmethod
    @abstractmethod
    def sample(state: AgentState, key: PRNGKey, *args, **kwargs) -> Any:
        """
        Selects the next action based on the current agent state.
        """

        pass

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        """
        Parameter space of the agent constructor in OpenAI Gym format.
        Type of returned value is required to be ``gym.spaces.Dict`` or ``None``.
        If ``None``, the user must provide all parameters manually.
        """

        return None

    @property
    def update_observation_space(self) -> gym.spaces.Space:
        """
        Observation space of the ``update`` method in OpenAI Gym format.
        If ``None``, the user must provide all parameters manually.
        """

        return None

    @property
    def sample_observation_space(self) -> gym.spaces.Space:
        """
        Observation space of the ``sample`` method in OpenAI Gym format.
        If ``None``, the user must provide all parameters manually.
        """

        return None

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Action space of the agent in OpenAI Gym format.
        """

        raise NotImplementedError()
