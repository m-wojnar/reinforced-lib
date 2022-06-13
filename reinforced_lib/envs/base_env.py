import abc
from typing import Callable, Dict, Tuple, Any

import gym.spaces

from envs.env_state import EnvState
from reinforced_lib.agents.agent_state import AgentState
from reinforced_lib.agents.base_agent import BaseAgent
from reinforced_lib.envs.utils import test_box, test_discrete, test_multi_binary, test_multi_discrete, FunctionInfo


class BaseEnv(abc.ABC):
    def __init__(self, agent: BaseAgent, agent_state: AgentState) -> None:
        """
        Container for functions of the environment and definition of action and state spaces.
        Provides transformation from environments functions and observation space to agents observation spaces.

        Fields
        ----------
        observation_space : gym.spaces.Space
            Parameters required by the environments 'act' function in OpenAI Gym format.
        action_space : gym.spaces.Space
            Action selected by the agent in OpenAI Gym format.

        Functions
        ----------
        update_space_transform : Callable
            Transformation to agents update_observation_space.
        sample_space_transform : Callable
            Transformation to agents sample_observation_space.
        reset : Callable
            Returns the environments initial state.
        act : Callable
            Updates the state of the environment after performing some action.
            Returns the action selected by agent based on the current environment state.

        Parameters
        ----------
        agent : BaseAgent
            Container for functions of the agent.
        agent_state : AgentState
            Container for the state of agent.
        """

        self._agent = agent
        self._agent_state = agent_state

        self._observation_space_functions: Dict[str, FunctionInfo] = {}
        self.update_space_transform = self._transform_spaces(self._agent.update_observation_space)
        self.sample_space_transform = self._transform_spaces(self._agent.sample_observation_space)

    def _transform_spaces(self, out_space: gym.spaces.Space) -> Callable:
        """
        Creates function that transforms environments functions and observation space to given space.

        Parameters
        ----------
        out_space : gym.spaces.Space
            Target space.

        Returns
        -------
        out : Callable
            Function that transforms environments functions and observation space to out_space.
        """

        if out_space is None:
            return lambda *args, **kwargs: None

        simple_types = {
            gym.spaces.Box: test_box,
            gym.spaces.Discrete: test_discrete,
            gym.spaces.MultiBinary: test_multi_binary,
            gym.spaces.MultiDiscrete: test_multi_discrete
        }

        if type(out_space) in simple_types:
            test_function = simple_types[type(out_space)]

            if test_function(self.observation_space, out_space):
                return self._simple_transform

            for space_function in dir(self):
                func = getattr(self, space_function)

                if not hasattr(func, 'function_info'):
                    continue

                if test_function(func.function_info.space_type, out_space):
                    return lambda *args, **kwargs: getattr(self, space_function)(*args, **kwargs)

        # TODO gym.spaces.Dict

        # TODO gym.spaces.Tuple

    @staticmethod
    def _simple_transform(*args, **kwargs) -> Any:
        assert len(args) + len(kwargs) == 1, 'Provided too many arguments!'

        if len(args) == 1:
            return args[0]
        else:
            first, *_ = kwargs.values()
            return first

    @property
    @abc.abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """
        Parameters required by the environments 'act' function in OpenAI Gym format.
        """
        pass

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.spaces.Space:
        """
        Action selected by the agent in OpenAI Gym format.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> EnvState:
        """
        Resets environment to initial state and returns the initial state.

        Returns
        -------
        out : EnvState
            Initial environment state.
        """
        pass

    @abc.abstractmethod
    def act(self, *args: Tuple, **kwargs: Dict) -> Any:
        """
        Updates the state of the environment after performing some action.
        Returns the action selected by agent based on the current environment state.

        Parameters
        ----------
        args : Tuple
            Environment observation space.
        kwargs : Dict
            Environment observation space.

        Returns
        -------
        out : Any
            Action selected by agent.
        """
        pass
