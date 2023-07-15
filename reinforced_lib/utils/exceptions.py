import gymnasium as gym


class NoAgentError(Exception):
    """
    Raised when no agent is specified.
    """

    def __str__(self) -> str:
        return 'No agent is specified.'


class NoExtensionError(Exception):
    """
    Raised when no extension is specified.
    """

    def __str__(self) -> str:
        return 'No extension is specified.'


class IncorrectTypeError(Exception):
    """
    Raised when provided class type is incorrect.

    Parameters
    ----------
    provided_type : type, optional
        Type provided by the user.
    expected_module : str, optional
        Name of the module that ``provided_type`` should match.
    """

    def __init__(self, provided_type: type = None, expected_module: str = None) -> None:
        self._provided_type = provided_type.__name__ if provided_type else 'Provided type'
        self._expected_module = expected_module if expected_module else ''

    def __str__(self) -> str:
        return f'{self._provided_type} is not a valid {self._expected_module} type.'


class IncorrectAgentTypeError(IncorrectTypeError):
    """
    Raised when provided agent does not inherit from the BaseAgent class.

    Parameters
    ----------
    provided_type : type
        Type provided by the user.
    """

    def __init__(self, provided_type: type) -> None:
        super().__init__(provided_type, 'agent')


class IncorrectExtensionTypeError(IncorrectTypeError):
    """
    Raised when provided extension does not inherit from the BaseExt class.

    Parameters
    ----------
    provided_type : type
        Type provided by the user.
    """

    def __init__(self, provided_type: type) -> None:
        super().__init__(provided_type, 'extension')


class IncorrectLoggerTypeError(IncorrectTypeError):
    """
    Raised when provided logger does not inherit from the BaseLogger class.

    Parameters
    ----------
    provided_type : type
        Type provided by the user.
    """

    def __init__(self, provided_type: type) -> None:
        super().__init__(provided_type, 'logger')


class ForbiddenOperationError(Exception):
    """
    Raised when the user performs a forbidden operation.
    """

    def __str__(self) -> str:
        return 'Forbidden operation.'


class ForbiddenAgentChangeError(ForbiddenOperationError):
    """
    Raised when the user changes the agent type after the first agent instance has been initialized.
    """

    def __str__(self) -> str:
        return 'Cannot change agent type after the first agent instance is initialized.'


class ForbiddenExtensionChangeError(ForbiddenOperationError):
    """
    Raised when the user changes the extension type after the first agent instance has been initialized.
    """

    def __str__(self) -> str:
        return 'Cannot change extension type after the first agent instance is initialized.'


class ForbiddenExtensionSetError(ForbiddenOperationError):
    """
    Raised when the user sets the extension type when ``no_ext_mode`` is enabled.
    """

    def __str__(self) -> str:
        return 'Cannot set extension type when \'no_ext_mode\' is enabled.'


class ForbiddenLoggerSetError(ForbiddenOperationError):
    """
    Raised when the user adds a new logger after the first step has been made.
    """

    def __str__(self) -> str:
        return 'Cannot add new loggers type after the first step has been made.'


class IncorrectSpaceError(Exception):
    """
    Raised when an unknown space is provided, for example a custom Gymnasium space.
    """

    def __str__(self) -> str:
        return 'Cannot find corresponding Gymnasium space.'


class UnimplementedSpaceError(Exception):
    """
    Raised when an observation space is required but not implemented.
    """

    def __str__(self) -> str:
        return 'Appropriate observation space is not implemented.'


class IncompatibleSpacesError(Exception):
    """
    Raised when the observation spaces of two different modules are not compatible.

    Parameters
    ----------
    ext_space : gym.spaces.Space
        Observation space of the extension.
    agent_space : gym.spaces.Space
        Observation space of the agent.
    """

    def __init__(self, ext_space: gym.spaces.Space, agent_space: gym.spaces.Space) -> None:
        self._ext_space = ext_space
        self._agent_space = agent_space

    def __str__(self) -> str:
        return f'Agents space of type {self._agent_space} is not compatible ' \
               f'with extension space of type {self._ext_space}.'


class NoDefaultParameterError(Exception):
    """
    Raised when the extension does not define a default parameter value for the agent.

    Parameters
    ----------
    extension_type : type
        Type of the used extension.
    parameter_name : str
        Name of the missing parameter.
    parameter_type : gym.spaces.Space
        Type of the missing parameter.
    """

    def __init__(self, extension_type: type, parameter_name: str, parameter_type: gym.spaces.Space) -> None:
        self._extension_name = extension_type.__name__
        self._parameter_name = parameter_name
        self._parameter_type = parameter_type

    def __str__(self) -> str:
        return f'Extension {self._extension_name} does not provide parameter ' \
               f'{self._parameter_name} of type {self._parameter_type}.'


class UnsupportedLogTypeError(Exception):
    """
    Raised when the user logs values that are not supported by the logger.

    Parameters
    ----------
    logger_type : type
        Type of the used logger.
    log_type : type
        Type of the logged value.
    """

    def __init__(self, logger_type: type, log_type: type) -> None:
        self._logger_name = logger_type.__name__
        self._log_name = log_type.__name__

    def __str__(self) -> str:
        return f'Logger {self._logger_name} does not support logging {self._log_name}.'


class IncorrectSourceTypeError(IncorrectTypeError):
    """
    Raised when the provided source is not a correct source type (i.e., ``Union[Tuple[str, SourceType], str]``).

    Parameters
    ----------
    provided_type : type
        Type provided by the user.
    """

    def __init__(self, provided_type: type) -> None:
        super().__init__(provided_type, 'source')


class UnsupportedCustomLogsError(Exception):
    """
    Raised when the user tries to log custom values with a logger that does not support custom logging.

    Parameters
    ----------
    logger_type : type
        Type of the used logger.
    """

    def __init__(self, logger_type: type) -> None:
        self._logger_name = logger_type.__name__

    def __str__(self) -> str:
        return f'Logger {self._logger_name} does not support custom logging.'
