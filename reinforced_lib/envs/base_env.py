from abc import abstractmethod, ABC
from functools import partial
from typing import Callable, Dict, Tuple, Any, List

import gym.spaces

from reinforced_lib.agents.agent_state import AgentState
from reinforced_lib.agents.base_agent import BaseAgent
from reinforced_lib.envs.env_state import EnvState
from reinforced_lib.envs.utils import test_box, test_discrete, test_multi_binary, test_multi_discrete, FunctionInfo
from reinforced_lib.utils.exceptions import IllegalSpaceError, IncompatibleSpacesError


class BaseEnv(ABC):
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
        update_space : Callable
            Transformation to agents update_observation_space.
        sample_space : Callable
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

        self._observation_space_functions: Dict[str, Callable] = {}

        for name in dir(self):
            obj = getattr(self, name)

            if hasattr(obj, 'function_info'):
                self._observation_space_functions[obj.function_info.parameter_name] = obj

        self.update_space = self._transform_spaces(self.observation_space, self._agent.update_observation_space)
        self.sample_space = self._transform_spaces(self.observation_space, self._agent.sample_observation_space)

    def _transform_spaces(self, in_space: gym.spaces.Space, out_space: gym.spaces.Space) -> Callable:
        """
        Creates function that transforms environments functions and observation space to given space.

        Parameters
        ----------
        in_space : gym.spaces.Space
            Source space.
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
            if type(in_space) not in simple_types:
                raise IncompatibleSpacesError(in_space, out_space)

            test_function = simple_types[type(out_space)]

            if test_function(in_space, out_space):
                return self._simple_transform

            for parameter_function in self._observation_space_functions.values():
                func_spec = parameter_function.function_info.space_type

                if func_spec is None or test_function(func_spec, out_space):
                    return lambda *args, **kwargs: parameter_function(*args, **kwargs)

        if isinstance(out_space, gym.spaces.Dict):
            if not isinstance(in_space, gym.spaces.Dict):
                raise IncompatibleSpacesError(in_space, out_space)

            parameters: Dict[str, Callable] = {}

            for name, space in out_space.spaces.items():
                if name in in_space.spaces:
                    if type(space) not in simple_types:
                        parameters[name] = self._transform_spaces(in_space[name], space)
                    elif simple_types[type(space)](in_space[name], space):
                        parameters[name] = lambda *args, **kwargs: kwargs[name]
                    else:
                        raise IncompatibleSpacesError(space, in_space)
                elif name in self._observation_space_functions:
                    func_space = self._observation_space_functions[name].function_info.space_type

                    if func_space is None or simple_types[type(space)](func_space, space):
                        parameters[name] = self._observation_space_functions[name]
                    else:
                        raise IncompatibleSpacesError(space, func_space)
                else:
                    raise IncompatibleSpacesError(space, in_space)

            return partial(self._dict_transform, parameters)

        if isinstance(out_space, gym.spaces.Tuple):
            if not isinstance(in_space, gym.spaces.Tuple) or len(in_space.spaces) != len(out_space.spaces):
                raise IncompatibleSpacesError(in_space, out_space)

            parameters: List[Callable] = []

            for i, (agent_space, env_space) in enumerate(zip(in_space.spaces, out_space.spaces)):
                if type(agent_space) in simple_types:
                    if simple_types[type(agent_space)](env_space, agent_space):
                        parameters.append(lambda *args, **kwargs: args[i])
                    else:
                        raise IncompatibleSpacesError(agent_space, in_space)
                else:
                    parameters.append(self._transform_spaces(env_space, agent_space))

            return partial(self._tuple_transform, parameters)

        raise IllegalSpaceError()

    @staticmethod
    def _simple_transform(*args: Tuple, **kwargs: Dict) -> Any:
        # TODO fix "leaf" methods

        assert len(args) + len(kwargs) == 1, 'Provided too many arguments!'

        if len(args) == 1:
            return args[0]
        else:
            first, *_ = kwargs.values()
            return first

    @staticmethod
    def _dict_transform(parameters: Dict[str, Callable], *args: Tuple, **kwargs: Dict) -> Dict:
        return {name: func(args, kwargs) for name, func in parameters.items()}

    @staticmethod
    def _tuple_transform(parameters: List[Callable], *args: Tuple, **kwargs: Dict) -> Tuple:
        return tuple(func(args, kwargs) for func in parameters)

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """
        Parameters required by the environments 'act' function in OpenAI Gym format.
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        """
        Action selected by the agent in OpenAI Gym format.
        """
        pass

    @abstractmethod
    def reset(self) -> EnvState:
        """
        Resets environment to initial state and returns the initial state.

        Returns
        -------
        out : EnvState
            Initial environment state.
        """
        pass

    @abstractmethod
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
