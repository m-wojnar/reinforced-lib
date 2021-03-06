from typing import Callable, NamedTuple

import gym.spaces
import numpy as np


class ObservationInfo(NamedTuple):
    """
    Description of the function that provides one of the values from the observation space.

    Attributes
    ----------
    name : str
        Name of the provided value.
    type : gym.spaces.Space
        Type of the provided value in OpenAI Gym format.
    """

    name: str
    type: gym.spaces.Space


class ParameterInfo(NamedTuple):
    """
    Description of the function that provides one of the parameters of the agent constructor.

    Attributes
    ----------
    name : str
        Name of the provided parameter.
    type : gym.spaces.Space
        Type of the provided parameter in OpenAI Gym format.
    """

    name: str
    type: gym.spaces.Space


def observation(observation_name: str = None, observation_type: gym.spaces.Space = None) -> Callable:
    """
    Decorator used to annotate functions that provide one of the values from the observation space.

    Parameters
    ----------
    observation_name : str, optional
        Name of the provided value.
    observation_type : gym.spaces.Space, optional
        Type of the provided value in OpenAI Gym format.

    Returns
    -------
    func : Callable
        Function that returns provided value.
    """

    def decorator(function):
        name = observation_name if observation_name is not None else function.__name__
        function.observation_info = ObservationInfo(name, observation_type)
        return function

    return decorator


def parameter(parameter_name: str = None, parameter_type: gym.spaces.Space = None) -> Callable:
    """
    Decorator used to annotate functions that provide one of the parameters of the agent constructor.

    Parameters
    ----------
    parameter_name : str, optional
        Name of the provided parameter.
    parameter_type : gym.spaces.Space, optional
        Type of the provided parameter in OpenAI Gym format.

    Returns
    -------
    func : Callable
        Function that returns provided parameter.
    """

    def decorator(function):
        name = parameter_name if parameter_name is not None else function.__name__
        function.parameter_info = ParameterInfo(name, parameter_type)
        return function

    return decorator


def test_box(a: gym.spaces.Space, b: gym.spaces.Box) -> bool:
    """
    Tests if space 'a' is identical to gym.space.Box space 'b'.

    Parameters
    ----------
    a : gym.spaces.Space
        Space 'a'.
    b : gym.spaces.Box
        Box space 'b'.

    Returns
    -------
    identical : bool
        Result of the comparison.
    """

    return isinstance(a, gym.spaces.Box) and \
        np.array_equal(a.low, b.low) and \
        np.array_equal(a.high, b.high) and \
        a.shape == b.shape and \
        a.dtype == b.dtype


def test_discrete(a: gym.spaces.Space, b: gym.spaces.Discrete) -> bool:
    """
    Tests if space 'a' is identical to gym.space.Discrete space 'b'.

    Parameters
    ----------
    a : gym.spaces.Space
        Space 'a'.
    b : gym.spaces.Discrete
        Discrete space 'b'.

    Returns
    -------
    identical : bool
        Result of the comparison.
    """

    return isinstance(a, gym.spaces.Discrete) and \
        a.n == b.n and \
        a.start == b.start


def test_multi_binary(a: gym.spaces.Space, b: gym.spaces.MultiBinary) -> bool:
    """
    Tests if space 'a' is identical to gym.space.MultiBinary space 'b'.

    Parameters
    ----------
    a : gym.spaces.Space
        Space 'a'.
    b : gym.spaces.MultiBinary
        MultiBinary space 'b'.

    Returns
    -------
    identical : bool
        Result of the comparison.
    """

    return isinstance(a, gym.spaces.MultiBinary) and \
        np.array_equal(a.n, b.n)


def test_multi_discrete(a: gym.spaces.Space, b: gym.spaces.MultiDiscrete) -> bool:
    """
    Tests if space 'a' is identical to gym.space.MultiDiscrete space 'b'.

    Parameters
    ----------
    a : gym.spaces.Space
        Space 'a'.
    b : gym.spaces.MultiDiscrete
        MultiDiscrete space 'b'.

    Returns
    -------
    identical : bool
        Result of the comparison.
    """

    return isinstance(a, gym.spaces.MultiDiscrete) and \
        np.array_equal(a.nvec, b.nvec) and \
        a.dtype == b.dtype


def test_space(a: gym.spaces.Space, b: gym.spaces.Space) -> bool:
    """
    Tests if space 'a' is identical to space 'b'.

    Parameters
    ----------
    a : gym.spaces.Space
        Space 'a'.
    b : gym.spaces.Space
        Space 'b'.

    Returns
    -------
    identical : bool
        Result of the comparison.
    """

    return a.shape == b.shape and \
        a.dtype == b.dtype
