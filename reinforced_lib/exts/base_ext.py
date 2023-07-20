from abc import ABC, abstractmethod
from functools import partial
import inspect
from typing import Union

import gymnasium as gym

from reinforced_lib.exts.utils import *
from reinforced_lib.utils.exceptions import IncorrectSpaceError, IncompatibleSpacesError, NoDefaultParameterError


class BaseExt(ABC):
    """
    Container for domain-specific knowledge and functions for a given environment. Provides the transformation
    from the raw observations to the agent update and sample spaces. Stores the default argument values for
    agent initialization.
    """

    def __init__(self) -> None:
        self._observation_functions: dict[str, Callable] = {}
        self._parameter_functions: dict[str, Callable] = {}

        self._add_action_to_observations = False

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
        Basic observations of the environment in Gymnasium format.
        """

        pass

    def get_agent_params(
            self,
            agent_type: type = None,
            agent_parameter_space: gym.spaces.Dict = None,
            user_parameters: dict[str, any] = None
    ) -> dict[str, any]:
        """
        Composes agent initialization arguments from values passed by the user and default values stored in the
        parameter functions. Returns a dictionary with the parameters matching the agent parameters space.

        Parameters
        ----------
        agent_type : type, optional
            Type of the selected agent.
        agent_parameter_space : gym.spaces.Dict, optional
            Parameters required by the agents' constructor in Gymnasium format.
        user_parameters : dict, optional
            Parameters provided by the user.

        Returns
        -------
        dict
            Dictionary with the initialization parameters for the agent.
        """

        parameters = user_parameters if user_parameters else {}

        if agent_parameter_space is None:
            return parameters

        default_parameters = set()

        if agent_type is not None:
            for key, value in inspect.signature(agent_type.__init__).parameters.items():
                if value.default != inspect._empty:
                    default_parameters.add(key)

        for name, space in agent_parameter_space.spaces.items():
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
        Creates functions that transform raw observations and values provided by the observation functions
        to the agent update and sample spaces.

        Parameters
        ----------
        agent_update_space : gym.spaces.Space, optional
            Observations required by the agent ``update`` function in Gymnasium format.
        agent_sample_space : gym.spaces.Space, optional
            Observations required by the agent ``sample`` function in Gymnasium format.
        """

        if 'action' not in self._observation_functions and \
                isinstance(self.observation_space, gym.spaces.Dict) and \
                'action' not in self.observation_space:

            if 'action' in agent_update_space.spaces:
                self.observation_space['action'] = agent_update_space['action']
            if 'action' in agent_sample_space.spaces:
                self.observation_space['action'] = agent_sample_space['action']

            self._add_action_to_observations = True

        self._update_space_transform = self._transform_spaces(self.observation_space, agent_update_space)
        self._sample_space_transform = self._transform_spaces(self.observation_space, agent_sample_space)

    def _transform_spaces(
            self,
            in_space: gym.spaces.Space,
            out_space: gym.spaces.Space,
            accessor: Union[str, int] = None
    ) -> Callable:
        """
        Creates function that transforms environment observations and values provided by the observation
        functions to a given space. If the ``out_space`` is not defined, returns observations unchanged.

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
        Callable
            Function that transforms values from ``in_space`` to ``out_space``.
        """

        if out_space is None:
            return lambda *args, **kwargs: None

        simple_types = {
            gym.spaces.Box: test_box,
            gym.spaces.Discrete: test_discrete,
            gym.spaces.MultiBinary: test_multi_binary,
            gym.spaces.MultiDiscrete: test_multi_discrete,
            gym.spaces.Sequence: test_sequence,
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

            observations: dict[str, Callable] = {}

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

            observations: list[Callable] = []

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
    def _get_nested_args(accessor: Union[str, int], *args, **kwargs) -> tuple[tuple, dict]:
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

            if isinstance(arguments, dict):
                return tuple(), arguments
            else:
                return arguments, {}

        return args, kwargs

    def _simple_transform(self, accessor: Union[str, int], *args, **kwargs) -> any:
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
        any
            Selected observation from the extension observation space.
        """

        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)

        if len(args) > 0:
            return args[0]
        else:
            first, *_ = kwargs.values()
            return first

    def _function_transform(self, func: Callable, name: str, accessor: Union[str, int], *args, **kwargs) -> any:
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
        any
            Selected observation.
        """

        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)

        if name in kwargs:
            return kwargs[name]
        else:
            return func(*args, **kwargs)

    def _dict_transform(self, observations: dict[str, Callable], accessor: Union[str, int], *args, **kwargs) -> dict:
        """
        Returns a dictionary filled with appropriate environment observations and values provided by
        the observation functions.

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
        dict
            Dictionary with functions providing observations.
        """

        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)
        return {name: func(*args, **kwargs) for name, func in observations.items()}

    def _tuple_transform(self, observations: list[Callable], accessor: Union[str, int], *args, **kwargs) -> tuple:
        """
        Returns a tuple filled with appropriate environment observations and values provided by
        the observation functions.

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
        tuple
            Tuple with functions providing observations.
        """

        args, kwargs = self._get_nested_args(accessor, *args, **kwargs)
        return tuple(func(*args, **kwargs) for func in observations)

    def transform(self, *args, action: any = None, **kwargs) -> tuple[any, any]:
        """
        Transforms raw observations and values provided by the observation functions to the agent observation
        and sample spaces. Provides the last action selected by the agent if it is required by the agent.

        Parameters
        ----------
        *args : tuple
            Environment observations.
        action : any
            The last action selected by the agent.
        **kwargs : dict
            Environment observations.

        Returns
        -------
        tuple[any, any]
            Agent update and sample observations.
        """

        if self._add_action_to_observations:
            kwargs['action'] = action

        return self._update_space_transform(*args, **kwargs), self._sample_space_transform(*args, **kwargs)
