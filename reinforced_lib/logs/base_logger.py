from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

from chex import Array, Scalar

from reinforced_lib.utils.exceptions import UnsupportedLogTypeError


class SourceType(Enum):
    OBSERVATION = 0
    STATE = 1
    METRIC = 2


Source = Union[Tuple[str, SourceType], str]


class BaseLogger(ABC):
    """
    Container for functions of the logger. Provides simple interface for defining custom loggers.
    """

    def __init__(self, **kwargs):
        pass

    def init(self, sources: List[Source]) -> None:
        """
        Initializes logger given the list of all sources.

        Parameters
        ----------
        sources : list[Source]
            List containing all sources for the logger.
        """

        pass

    def finish(self) -> None:
        """
        Used to finalize loggers work, for example save data ot show plots.
        """

        pass

    def log_scalar(self, source: Source, value: Scalar) -> None:
        """
        Method of the logger interface used for logging scalar values.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : float
            Scalar to log.
        """

        raise UnsupportedLogTypeError(type(self), type(value))

    def log_array(self, source: Source, value: Array) -> None:
        """
        Method of the logger interface used for logging arrays.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : array_like
            Array to log.
        """

        raise UnsupportedLogTypeError(type(self), type(value))

    def log_dict(self, source: Source, value: Dict) -> None:
        """
        Method of the logger interface used for logging dictionaries.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : dict
            Dictionary to log.
        """

        raise UnsupportedLogTypeError(type(self), type(value))

    def log_other(self, source: Source, value: Any) -> None:
        """
        Method of the logger interface used for logging other values.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : any
            Value of any type to log.
        """

        raise UnsupportedLogTypeError(type(self), type(value))

    @staticmethod
    def source_to_name(source: Source) -> str:
        """
        Converts source to a string name. If source is a string itself, it returns that string.
        Otherwise, it returns string in the format "name-sourcetype" (e.g. "action-metric").

        Parameters
        ----------
        source : Source
            Source of the logged value.
        """

        if isinstance(source, tuple):
            return f'{source[0]}-{source[1].name.lower()}'
        else:
            return source
