import gym.spaces


class IllegalSpaceError(Exception):
    """
    Exception is raised when unknown space is provided, for example custom OpenAI Gym space.
    """
    def __str__(self) -> str:
        return 'Cannot find corresponding OpenAI Gym space.'


class IncompatibleSpacesError(Exception):
    """
    Exception is raised when observation spaces of two different modules are not compatible.
    """

    def __init__(self, env_space: gym.spaces.Space, agent_space: gym.spaces.Space) -> None:
        self._env_space = env_space
        self._agent_space = agent_space

    def __str__(self) -> str:
        return f'Agents space of type {self._agent_space} is not compatible ' \
               f'with environment space of type {self._env_space}.'
