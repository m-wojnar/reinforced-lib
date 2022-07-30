from abc import ABC
from typing import Any, Dict, List

from chex import Array, Scalar


class BaseLogger(ABC):
    def __init__(self, **kwargs):
        pass

    def init(self, names: List[str]) -> None:
        pass

    def finish(self) -> None:
        pass

    def log_scalar(self, name: str, value: Scalar) -> None:
        pass

    def log_dict(self, name: str, value: Dict) -> None:
        pass

    def log_array(self, name: str, value: Array) -> None:
        pass

    def log_other(self, name: str, value: Any) -> None:
        pass
