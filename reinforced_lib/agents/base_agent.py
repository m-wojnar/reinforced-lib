from typing import NamedTuple, Callable

import gym


class BaseAgent(NamedTuple):
    """
    Container for functions of the agent.

    Fields
    ------
    init : Callable
        Creates and initializes state for the agent.
    update : Callable
        Updates the state of the agent after performing some action and receiving a reward.
    sample : Callable
        Selects next action based on current agent state.

    update_observation_space : gym.spaces.Space
        Parameters required by the agents 'update' function in OpenAI Gym format.
    sample_observation_space : gym.spaces.Space
        Parameters required by the agents 'sample' function in OpenAI Gym format.
    action_space : gym.spaces.Space
        Action returned by the agent in OpenAI Gym format.
    """

    init: Callable
    update: Callable
    sample: Callable

    update_observation_space: gym.spaces.Space
    sample_observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
