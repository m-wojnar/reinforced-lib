from abc import ABC, abstractmethod
from typing import Any, Tuple

import gym.spaces
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
    def sample(state: AgentState, key: PRNGKey, *args, **kwargs) -> Tuple[AgentState, Any]:
        """
        Selects the next action based on the current agent state.
        """

        pass

    @staticmethod
    @abstractmethod
    def parameters_space() -> gym.spaces.Dict:
        """
        Parameters space of the agent constructor in OpenAI Gym format.
        Type of returned value is required to be ``gym.spaces.Dict``.
        """

        pass

    @property
    @abstractmethod
    def update_observation_space(self) -> gym.spaces.Space:
        """
        Observation space of the ``update`` method in OpenAI Gym format.
        """

        pass

    @property
    @abstractmethod
    def sample_observation_space(self) -> gym.spaces.Space:
        """
        Observation space by the ``sample`` method in OpenAI Gym format.
        """

        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        """
        Action space of the agent in OpenAI Gym format.
        """

        pass
