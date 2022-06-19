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
    def init() -> AgentState:
        """
        Creates and initializes instance of the agent.
        """

        pass

    @staticmethod
    def update(state: AgentState, key: chex.PRNGKey, *args, **kwargs) -> AgentState:
        """
        Updates the state of the agent after performing some action and receiving a reward.
        """

        pass

    @staticmethod
    def sample(state: AgentState, key: chex.PRNGKey, *args, **kwargs) -> Tuple[AgentState, Any]:
        """
        Selects next action based on current agent state.
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
