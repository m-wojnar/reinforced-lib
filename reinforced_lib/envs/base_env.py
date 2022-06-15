from typing import NamedTuple, Callable

import gym.spaces


class BaseEnv(NamedTuple):
    """
    Container for functions of the environment.

    Fields
    ----------
    reset : Callable
        Returns the environments initial state.
    act : Callable
        Updates the state of the environment after performing some action. Returns
        a tuple of updated state and the reward (chex.Scalar) after taking that action
        in previous state
    
    action_space : gym.spaces.Space
        Parameters required by the environments 'act' function in OpenAI Gym format.
    state_space : gym.spaces.Space
        State of the environment in OpenAI Gym format.
    """

    reset : Callable
    act : Callable

    action_space : gym.spaces.Space
    state_space : gym.spaces.Space
