from abc import ABC, abstractmethod
from functools import partial
import inspect
from typing import Any, Callable, Dict, List, Tuple, Union

import gym.spaces

from reinforced_lib.exts.utils import test_box, test_discrete, test_multi_binary, test_multi_discrete, test_space
from reinforced_lib.utils.exceptions import IncorrectSpaceError, IncompatibleSpacesError, NoDefaultParameterError


class BaseExt(ABC):
    """
    Container for domain-specific knowledge and functions for a given environment. Provides transformation
    from observation functions and observation space to agents update and sample spaces. Stores default
    argument values for agents initialization.
    """

    def __init__(self) -> None:
        self._observation_functions: Dict[str, Callable] = {}
        self._parameter_functions: Dict[str, Callable] = {}

        for name in dir(self):
            obj = getattr(self, name)

            if hasattr(obj, 'observation_info'):
                self._observation_functions[obj.observation_info.name] = obj

            if hasattr(obj, 'parameter_info'):
                self._parameter_functions[obj.parameter_info.name] = obj

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """
        Basic observations of the environment in the OpenAI Gym format.
        """

        pass

    def get_agent_params(
            self,
            agent_type: type = None,
            agent_parameters_space: gym.spaces.Dict = None,
            user_parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Composes agent initialization parameters from the parameters passed by the user and default values defined
        in parameter functions. Returns dictionary with parameters fitting the agents parameters space.

        Parameters
        ----------
        agent_type : type, optional
            Type of selected agent.
        agent_parameters_space : gym.spaces.Dict, optional
            Parameters required by the agents' constructor in the OpenAI Gym format.
        user_parameters : dict, optional
            Parameters provided by the user.

        Returns
        -------
        parameters : dict
            Dictionary with constructor parameters for the agent.
        """

        if agent_parameters_space is None:
            return {}

        default_parameters = set()

        if agent_type is not None:
            for key, value in inspect.signature(agent_type.__init__).parameters.items():
                if value.default != inspect._empty:
                    default_parameters.add(key)

        parameters = user_parameters if user_parameters else {}

        for name, space in agent_parameters_space.spaces.items():
            if name in parameters:
                continue

            if name not in self._parameter_functions:
                if name in default_parameters:
                    continue

                raise NoDefaultParameterError(type(self), name, space)

            func = self._parameter_functions[name]
            func_space = func.parameter_info.type

            if space is None or type(space) == type(func_space):
                parameters[name] = func()
            else:
                raise IncompatibleSpacesError(func_space, space)

        return parameters

    def setup_transformations(
            self,
            agent_update_space: gym.spaces.Space = None,
            agent_sample_space: gym.spaces.Space = None
    ) -> None:
        """
        Create functions that transform environment observations and values provided by observation functions to agents
        update and sample spaces values.

        Parameters
        ----------
        agent_update_space : gym.spaces.Space, optional
            Observations required by the agents 'update' function in the OpenAI Gym format.
        agent_sample_space : gym.spaces.Space, optional
            Observations required by the agents 'sample' function in the OpenAI Gym format.
        """

        self._update_space_transform = self._transform_spaces(self.observation_space, agent_update_space)
        self._sample_space_transform = self._transform_spaces(self.observation_space, agent_sample_space)

    def _transform_spaces(
            self,
            in_space: gym.spaces.Space,
            out_space: gym.spaces.Space,
            accessor: Union[str, int] = None
    ) -> Callable:
        """
        Creates function that transforms environment observations and values provided by observation functions to
        a given space values.

        Parameters
        ----------
        in_space : gym.spaces.Space
            Source space.
        out_space : gym.spaces.Space
            Target space.
        accessor : str or int, optional
            Path to nested observations.

        Returns
        -------
        func : Callable
            Function that transforms values from in_space to out_space.
        """

        if out_space is None:
            return lambda *args, **kwargs: None

        simple_types = {
            gym.spaces.Box: test_box,
            gym.spaces.Discrete: test_discrete,
            gym.spaces.MultiBinary: test_multi_binary,
            gym.spaces.MultiDiscrete: test_multi_discrete,
            gym.spaces.Space: test_space
        }

        if type(out_space) in simple_types:
            if type(in_space) not in simple_types:
                raise IncompatibleSpacesError(in_space, out_space)

            test_function = simple_types[type(out_space)]

            if test_function(in_space, out_space):
                return partial(self._simple_transform, accessor)

            for observation_function in self._observation_functions.values():
                func_space = observation_function.observation_info.type

                if func_space is None or test_function(func_space, out_space):
                    return observation_function

        if isinstance(out_space, gym.spaces.Dict):
            if not isinstance(in_space, gym.spaces.Dict):
                raise IncompatibleSpacesError(in_space, out_space)

            observations: Dict[str, Callable] = {}

            for name, space in out_space.spaces.items():
                if name in in_space.spaces:
                    if type(space) not in simple_types:
                        observations[name] = self._transform_spaces(in_space[name], space, name)
                    elif simple_types[type(space)](in_space[name], space):
                        observations[name] = partial(lambda inner_name, *args, **kwargs: kwargs[inner_name], name)
                    else:
                        raise IncompatibleSpacesError(in_space, space)
                elif name in self._observation_functions:
                    func_space = self._observation_functions[name].observation_info.type

                    if func_space is None or simple_types[type(space)](func_space, space):
                        observations[name] = partial(
                            lambda func, inner_name, inner_accessor, *args, **kwargs:
                            self._function_transform(func, inner_name, inner_accessor, *args, **kwargs),
                            self._observation_functions[name], name, accessor
                        )
                    else:
                        raise IncompatibleSpacesError(func_space, space)
                else:
                    raise IncompatibleSpacesError(in_space, space)

            return partial(self._dict_transform, observations, accessor)

        if isinstance(out_space, gym.spaces.Tuple):
            if not isinstance(in_space, gym.spaces.Tuple) or len(in_space.spaces) != len(out_space.spaces):
                raise IncompatibleSpacesError(in_space, out_space)

            observations: List[Callable] = []

            for i, (agent_space, ext_space) in enumerate(zip(in_space.spaces, out_space.spaces)):
                if type(agent_space) not in simple_types:
                    observations.append(self._transform_spaces(ext_space, agent_space, i))
                elif simple_types[type(agent_space)](ext_space, agent_space):
                    observations.append(partial(lambda inner_i, *args, **kwargs: args[inner_i], i))
                else:
                    raise IncompatibleSpacesError(agent_space, in_space)

            return partial(self._tuple_transform, observations, accessor)

        raise IncorrectSpaceError()

    @staticmethod
    def _get_nested_args(accessor: Union[str, int], *args, **kwargs) -> Tuple[Tuple, Dict]:
        """
        Selects the appropriate nested args or kwargs.

        Parameters
        ----------
        accessor : str or int
            Path to nested observations.
        *args : tuple
            Environment observations.
        **kwargs : dict
            Environment observations.

        Returns
        -------
        tuple[tuple, dict]
            Args and kwargs.
        """

        if accessor is not None:
            if isinstance(accessor, int):
                arguments = args[accessor]
            else:
                arguments = kwargs[accessor]

            if isinstance(arguments, Dict):
                return tuple(), arguments
            else:
                return arguments, {}

        return args, kwargs

    def _simple_transform(self, accessor: Union[str, int], *args, **kwargs) -> Any:
        """
        Returns the appropriate observation from environment observations.

        Parameters
        ----------
        accessor : str or int
            Path to nested observations.
        *args : tuple
            Environment observations.
        **kwargs : dict
            Environment observations.

        Returns
        -------
        observation : any
            Selected observation from extension observation space.
        """

        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)

        if len(args) > 0:
            return args[0]
        else:
            first, *_ = kwargs.values()
            return first

    def _function_transform(self, func: Callable, name: str, accessor: Union[str, int], *args, **kwargs) -> Any:
        """
        Returns the appropriate observation from the observation function or from environment observations
        if present.

        Parameters
        ----------
        func : Callable
            Function that returns selected observation.
        name : str
            Name of the selected observation.
        accessor : str or int
            Path to nested observations.
        *args : tuple
            Environment observations.
        **kwargs : dict
            Environment observations.

        Returns
        -------
        observation : any
            Selected observation.
        """

        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)

        if name in kwargs:
            return kwargs[name]
        else:
            return func(*args, **kwargs)

    def _dict_transform(self, observations: Dict[str, Callable], accessor: Union[str, int], *args, **kwargs) -> Dict:
        """
        Returns a dictionary filled with appropriate environment observations and values provided by
        observation functions.

        Parameters
        ----------
        observations : dict[str, Callable]
            Dictionary with observation names and functions that provide according observations.
        accessor : str or int
            Path to nested observations.
        *args : tuple
            Environment observations.
        **kwargs : dict
            Environment observations.

        Returns
        -------
        observations : dict
            Dictionary with functions providing observations.
        """

        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)
        return {name: func(*args, **kwargs) for name, func in observations.items()}

    def _tuple_transform(self, observations: List[Callable], accessor: Union[str, int], *args, **kwargs) -> Tuple:
        """
        Returns a tuple filled with appropriate environment observations and values provided by observation functions.

        Parameters
        ----------
        observations : list[Callable]
            List with functions that provide selected observations.
        accessor : str or int
            Path to nested observations.
        *args : tuple
            Environment observations.
        **kwargs : dict
            Environment observations.

        Returns
        -------
        observations : tuple
            Tuple with functions providing observations.
        """

        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)
        return tuple(func(*args, **kwargs) for func in observations)

    def transform(self, *args, **kwargs) -> Tuple[Any, Any]:
        """
        Transforms environment observations and values provided by observation functions to agent observation
        and sample spaces values.

        Parameters
        ----------
        *args : tuple
            Environment observations.
        **kwargs : dict
            Environment observations.

        Returns
        -------
        observations : tuple[any, any]
            Agents update and sample observations.
        """

        return self._update_space_transform(*args, **kwargs), self._sample_space_transform(*args, **kwargs)
