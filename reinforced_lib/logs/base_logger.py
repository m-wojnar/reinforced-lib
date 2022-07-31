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
    def __init__(self, **kwargs):
        pass

    def init(self, sources: List[Source]) -> None:
        pass

    def finish(self) -> None:
        pass

    def log_scalar(self, source: Source, value: Scalar) -> None:
        raise UnsupportedLogTypeError(type(self), type(value))

    def log_dict(self, source: Source, value: Dict) -> None:
        raise UnsupportedLogTypeError(type(self), type(value))

    def log_array(self, source: Source, value: Array) -> None:
        raise UnsupportedLogTypeError(type(self), type(value))

    def log_other(self, source: Source, value: Any) -> None:
        raise UnsupportedLogTypeError(type(self), type(value))

    @staticmethod
    def source_to_name(source: Source) -> str:
        if isinstance(source, tuple):
            return f'{source[0]}-{source[1].name.lower()}'
        else:
            return source
