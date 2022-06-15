from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

import gym.spaces

from reinforced_lib.envs.utils import test_box, test_discrete, test_multi_binary, test_multi_discrete
from reinforced_lib.utils.exceptions import IllegalSpaceError, IncompatibleSpacesError


class BaseEnv(ABC):
    def __init__(
            self,
            agent_update_space: gym.spaces.Space = None,
            agent_sample_space: gym.spaces.Space = None
    ) -> None:
        """
        Container for domain-specific knowledge and functions for a given environment. Provides transformation
        from environment functions and observation space to agents observation and sample spaces.

        Parameters
        ----------
        agent_update_space : gym.spaces.Space
            Observations required by the agents 'update' function in OpenAI Gym format.
        agent_sample_space : gym.spaces.Space
            Observations required by the agents 'sample' function in OpenAI Gym format.
        """

        self._observation_functions: Dict[str, Callable] = {}

        for name in dir(self):
            obj = getattr(self, name)

            if hasattr(obj, 'function_info'):
                self._observation_functions[obj.function_info.observation_name] = obj

        self._update_space_transform = self._transform_spaces(self.observation_space, agent_update_space)
        self._sample_space_transform = self._transform_spaces(self.observation_space, agent_sample_space)

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """
        Observations taken by the 'transform' function in OpenAI Gym format.
        """

        pass

    def _transform_spaces(
            self,
            in_space: gym.spaces.Space,
            out_space: gym.spaces.Space,
            accessor: Union[str, int] = None
    ) -> Callable:
        """
        Creates function that transforms environment functions and observation space to given space.

        Parameters
        ----------
        in_space : gym.spaces.Space
            Source space.
        out_space : gym.spaces.Space
            Target space.
        accessor : Union[str, int]
            Path to nested observations.

        Returns
        -------
        out : Callable
            Function that transforms environment functions and in_space to out_space.
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
                return partial(self._simple_transform, accessor)

            for observation_function in self._observation_functions.values():
                func_spec = observation_function.function_info.observation_type

                if func_spec is None or test_function(func_spec, out_space):
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
                        raise IncompatibleSpacesError(space, in_space)
                elif name in self._observation_functions:
                    func_space = self._observation_functions[name].function_info.observation_type

                    if func_space is None or simple_types[type(space)](func_space, space):
                        observations[name] = partial(
                            lambda func, inner_name, inner_accessor, *args, **kwargs:
                            self._function_transform(func, inner_name, inner_accessor, *args, **kwargs),
                            self._observation_functions[name], name, accessor
                        )
                    else:
                        raise IncompatibleSpacesError(space, func_space)
                else:
                    raise IncompatibleSpacesError(space, in_space)

            return partial(self._dict_transform, observations, accessor)

        if isinstance(out_space, gym.spaces.Tuple):
            if not isinstance(in_space, gym.spaces.Tuple) or len(in_space.spaces) != len(out_space.spaces):
                raise IncompatibleSpacesError(in_space, out_space)

            observations: List[Callable] = []

            for i, (agent_space, env_space) in enumerate(zip(in_space.spaces, out_space.spaces)):
                if type(agent_space) not in simple_types:
                    observations.append(self._transform_spaces(env_space, agent_space, i))
                elif simple_types[type(agent_space)](env_space, agent_space):
                    observations.append(partial(lambda inner_i, *args, **kwargs: args[inner_i], i))
                else:
                    raise IncompatibleSpacesError(agent_space, in_space)

            return partial(self._tuple_transform, observations, accessor)

        raise IllegalSpaceError()

    @staticmethod
    def _get_nested_args(accessor: Union[str, int], *args, **kwargs) -> Tuple[Tuple, Dict]:
        """
        Selects the appropriate nested args and kwargs.

        Parameters
        ----------
        accessor : Union[str, int]
            Path to nested observations.
        args : Tuple
            Environment observation space.
        kwargs : Dict
            Environment observation space.

        Returns
        -------
        out : Tuple[Tuple, Dict]
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
        Returns the appropriate observation from environment observation space.

        Parameters
        ----------
        accessor : Union[str, int]
            Path to nested observations.
        args : Tuple
            Environment observation space.
        kwargs : Dict
            Environment observation space.

        Returns
        -------
        out : Any
            Selected observation from environment observation space.
        """

        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)

        if len(args) > 0:
            return args[0]
        else:
            first, *_ = kwargs.values()
            return first

    def _function_transform(self, func: Callable, name: str, accessor: Union[str, int], *args, **kwargs) -> Any:
        """
        Returns the appropriate observation from environment function (or from observation space if present).

        Parameters
        ----------
        func : Callable
            Function that returns selected observation.
        name : str
            Name of selected observation.
        accessor : Union[str, int]
            Path to nested observations.
        args : Tuple
            Environment observation space.
        kwargs : Dict
            Environment observation space.

        Returns
        -------
        out : Any
            Selected observation from environment function.
        """

        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)

        if name in kwargs:
            return kwargs[name]
        else:
            return func(*args, **kwargs)

    def _dict_transform(self, observations: Dict[str, Callable], accessor: Union[str, int], *args, **kwargs) -> Dict:
        """
        Returns a dictionary filled with appropriate observations from environment functions and observation space.
        
        Parameters
        ----------
        observations : Dict[str, Callable]
            Dictionary with observation names and functions that provide selected observations.
        accessor : Union[str, int]
            Path to nested observations.
        args : Tuple
            Environment observation space.
        kwargs : Dict
            Environment observation space.

        Returns
        -------
        out : Dict
            Dictionary with selected observations.
        """
        
        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)
        return {name: func(*args, **kwargs) for name, func in observations.items()}

    def _tuple_transform(self, observations: List[Callable], accessor: Union[str, int], *args, **kwargs) -> Tuple:
        """
        Returns a tuple filled with appropriate observations from environment functions and observation space.

        Parameters
        ----------
        observations : List[Callable]
            List with functions that provide selected observations.
        accessor : Union[str, int]
            Path to nested observations.
        args : Tuple
            Environment observation space.
        kwargs : Dict
            Environment observation space.

        Returns
        -------
        out : Tuple
            Tuple with selected observations.
        """

        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)
        return tuple(func(*args, **kwargs) for func in observations)

    def transform(self, *args, **kwargs) -> Tuple[Any, Any]:
        """
        Transforms environment functions and observation space to agents observation and sample spaces.

        Parameters
        ----------
        args : Tuple
            Environment observation space.
        kwargs : Dict
            Environment observation space.

        Returns
        -------
        out : Tuple[Any, Any]
            Agents observation and sample spaces.
        """

        return self._update_space_transform(*args, **kwargs), self._sample_space_transform(*args, **kwargs)
