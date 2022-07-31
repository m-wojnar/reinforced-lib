from typing import Any, Dict, List

from chex import Array, Scalar

from reinforced_lib.logs import BaseLogger, Source


class StdoutLogger(BaseLogger):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._values = {}
        self._names = []

    def init(self, sources: List[Source]) -> None:
        self._names = list(map(self.source_to_name, sources))

    def log_scalar(self, source: Source, value: Scalar) -> None:
        if (name := self.source_to_name(source)) in self._values:
            self._print()
        else:
            self._values[name] = value

    def log_array(self, source: Source, value: Array) -> None:
        if (name := self.source_to_name(source)) in self._values:
            self._print()
        else:
            self._values[name] = ' '.join(str(v) for v in value)

    def log_dict(self, source: Source, value: Dict) -> None:
        self.log_other(source, value)

    def log_other(self, source: Source, value: Any) -> None:
        if (name := self.source_to_name(source)) in self._values:
            self._print()
        else:
            self._values[name] = str(value)

    def _print(self) -> None:
        for name in self._names:
            if name in self._values:
                print(f'{name}: {self._values[name]}', end='\t')

        self._values = {}
        print()
