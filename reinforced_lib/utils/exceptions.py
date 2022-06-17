from typing import Any

import gym.spaces


class NoAgentError(Exception):
    """
    Raised when no agent is specified.
    """

    def __str__(self) -> str:
        return 'No agent is specified.'


class NoEnvironmentError(Exception):
    """
    Raised when no environment is specified.
    """

    def __str__(self) -> str:
        return 'No environment is specified.'


class IncorrectTypeError(Exception):
    """
    Raised when provided class type is incorrect.
    """

    def __init__(self, provided_type: type = None, expected: str = None) -> None:
        self._provided_type = provided_type.__name__ if provided_type else 'Provided type'
        self._expected = expected if expected else ''

    def __str__(self) -> str:
        return f'{self._provided_type} is not a valid {self._expected} type.'


class IncorrectAgentTypeError(IncorrectTypeError):
    """
    Raised when provided agent is not an agent class.
    """

    def __init__(self, provided_type: Any) -> None:
        super(provided_type, 'agent')


class IncorrectEnvironmentTypeError(IncorrectTypeError):
    """
    Raised when provided environment is not an environment class.
    """

    def __init__(self, provided_type: Any) -> None:
        super(provided_type, 'environment')
        
        
class ForbiddenOperationError(Exception):
    """
    Raised when user is trying to perform forbidden operation.
    """

    def __str__(self) -> str:
        return 'Forbidden operation.'


class ForbiddenAgentChangeError(ForbiddenOperationError):
    """
    Raised when user is trying to change the agent type after the first agent instance is initialized.
    """

    def __str__(self) -> str:
        return 'Cannot change agent type after the first agent instance is initialized.'


class ForbiddenEnvironmentChangeError(ForbiddenOperationError):
    """
    Raised when user is trying to change the environment type after the first agent instance is initialized.
    """

    def __str__(self) -> str:
        return 'Cannot change environment type after the first agent instance is initialized.'


class ForbiddenEnvironmentSetError(ForbiddenOperationError):
    """
    Raised when user is trying to set environment type when 'no_env_mode' is enabled.
    """

    def __str__(self) -> str:
        return 'Cannot set environment type when \'no_env_mode\' is enabled.'


class IncorrectSpaceError(Exception):
    """
    Raised when unknown space is provided, for example custom OpenAI Gym space.
    """

    def __str__(self) -> str:
        return 'Cannot find corresponding OpenAI Gym space.'


class IncompatibleSpacesError(Exception):
    """
    Raised when observation spaces of two different modules are not compatible.
    """

    def __init__(self, env_space: gym.spaces.Space, agent_space: gym.spaces.Space) -> None:
        self._env_space = env_space
        self._agent_space = agent_space

    def __str__(self) -> str:
        return f'Agents space of type {self._agent_space} is not compatible ' \
               f'with environment space of type {self._env_space}.'
