from typing import Callable, NamedTuple

import gymnasium as gym
import numpy as np


class ObservationInfo(NamedTuple):
    """
    Description of the observation function that provides one of the values from the agent observation space.

    Attributes
    ----------
    name : str
        Name of the provided observation.
    type : gym.spaces.Space
        Type of the provided value in Gymnasium format.
    """

    name: str
    type: gym.spaces.Space


class ParameterInfo(NamedTuple):
    """
    Description of the parameter function that provides one of the parameters of the agent constructor.

    Attributes
    ----------
    name : str
        Name of the provided parameter.
    type : gym.spaces.Space
        Type of the provided parameter in Gymnasium format.
    """

    name: str
    type: gym.spaces.Space


def observation(observation_name: str = None, observation_type: gym.spaces.Space = None) -> Callable:
    """
    Decorator used to annotate the observation functions.

    Parameters
    ----------
    observation_name : str, optional
        Name of the provided observation.
    observation_type : gym.spaces.Space, optional
        Type of the provided value in Gymnasium format.

    Returns
    -------
    Callable
        Function that returns the appropriate observation.
    """

    def decorator(function):
        name = observation_name if observation_name is not None else function.__name__
        function.observation_info = ObservationInfo(name, observation_type)
        return function

    return decorator


def parameter(parameter_name: str = None, parameter_type: gym.spaces.Space = None) -> Callable:
    """
    Decorator used to annotate the parameter functions.

    Parameters
    ----------
    parameter_name : str, optional
        Name of the provided parameter.
    parameter_type : gym.spaces.Space, optional
        Type of the provided parameter in Gymnasium format.

    Returns
    -------
    Callable
        Function that returns the appropriate parameter.
    """

    def decorator(function):
        name = parameter_name if parameter_name is not None else function.__name__
        function.parameter_info = ParameterInfo(name, parameter_type)
        return function

    return decorator


def test_box(a: gym.spaces.Space, b: gym.spaces.Box) -> bool:
    """
    Tests if the space ``a`` is identical to the gym.space.Box space ``b``.

    Parameters
    ----------
    a : gym.spaces.Space
        Space ``a``.
    b : gym.spaces.Box
        Box space ``b``.

    Returns
    -------
    bool
        Result of the comparison.
    """

    return isinstance(a, gym.spaces.Box) and \
        np.array_equal(a.low, b.low) and \
        np.array_equal(a.high, b.high) and \
        a.shape == b.shape and \
        a.dtype == b.dtype


def test_discrete(a: gym.spaces.Space, b: gym.spaces.Discrete) -> bool:
    """
    Tests if the space ``a`` is identical to the gym.space.Discrete space ``b``.

    Parameters
    ----------
    a : gym.spaces.Space
        Space ``a``.
    b : gym.spaces.Discrete
        Discrete space ``b``.

    Returns
    -------
    bool
        Result of the comparison.
    """

    return isinstance(a, gym.spaces.Discrete) and \
        a.n == b.n and \
        a.start == b.start


def test_multi_binary(a: gym.spaces.Space, b: gym.spaces.MultiBinary) -> bool:
    """
    Tests if the space ``a`` is identical to the gym.space.MultiBinary space ``b``.

    Parameters
    ----------
    a : gym.spaces.Space
        Space ``a``.
    b : gym.spaces.MultiBinary
        MultiBinary space ``b``.

    Returns
    -------
    bool
        Result of the comparison.
    """

    return isinstance(a, gym.spaces.MultiBinary) and \
        np.array_equal(a.n, b.n)


def test_multi_discrete(a: gym.spaces.Space, b: gym.spaces.MultiDiscrete) -> bool:
    """
    Tests if the space ``a`` is identical to the gym.space.MultiDiscrete space ``b``.

    Parameters
    ----------
    a : gym.spaces.Space
        Space ``a``.
    b : gym.spaces.MultiDiscrete
        MultiDiscrete space ``b``.

    Returns
    -------
    bool
        Result of the comparison.
    """

    return isinstance(a, gym.spaces.MultiDiscrete) and \
        np.array_equal(a.nvec, b.nvec) and \
        a.dtype == b.dtype


def test_sequence(a: gym.spaces.Space, b: gym.spaces.Sequence) -> bool:
    """
    Tests if the space ``a`` is identical to the gym.space.Sequence space ``b``.

    Parameters
    ----------
    a : gym.spaces.Space
        Space ``a``.
    b : gym.spaces.Sequence
        Sequence space ``b``.

    Returns
    -------
    bool
        Result of the comparison.
    """

    if not isinstance(a, gym.spaces.Sequence):
        return False

    if isinstance(b, gym.spaces.Box):
        return test_box(a, b)
    elif isinstance(b, gym.spaces.Discrete):
        return test_discrete(a, b)
    elif isinstance(b, gym.spaces.MultiBinary):
        return test_multi_binary(a, b)
    elif isinstance(b, gym.spaces.MultiDiscrete):
        return test_multi_discrete(a, b)
    else:
        return test_space(a, b)


def test_space(a: gym.spaces.Space, b: gym.spaces.Space) -> bool:
    """
    Tests if the space ``a`` is identical to the space ``b``.

    Parameters
    ----------
    a : gym.spaces.Space
        Space ``a``.
    b : gym.spaces.Space
        Space ``b``.

    Returns
    -------
    bool
        Result of the comparison.
    """

    return a.shape == b.shape and \
        a.dtype == b.dtype
