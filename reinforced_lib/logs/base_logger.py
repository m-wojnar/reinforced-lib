from abc import ABC
from typing import Any, Dict, List, Tuple, Union

from chex import Array, Scalar

Source = Union[Tuple[str, Any], str]


class BaseLogger(ABC):
    def __init__(self, **kwargs):
        pass

    def init(self, sources: List[Source]) -> None:
        pass

    def finish(self) -> None:
        pass

    def log_scalar(self, source: Source, value: Scalar) -> None:
        pass

    def log_dict(self, source: Source, value: Dict) -> None:
        pass

    def log_array(self, source: Source, value: Array) -> None:
        pass

    def log_other(self, source: Source, value: Any) -> None:
        pass
