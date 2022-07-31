from typing import Any, Dict, List

from chex import Array, Scalar

from reinforced_lib.logs import BaseLogger, Source


class CsvLogger(BaseLogger):
    def __init__(self, csv_path: str = 'output.csv', **kwargs) -> None:
        super().__init__(**kwargs)

        self._file = open(csv_path, 'w')

        self._columns_values = {}
        self._columns_names = []

    def init(self, sources: List[Source]) -> None:
        self._columns_names = list(map(self.source_to_name, sources))
        header = ','.join(self._columns_names)
        self._file.write(f'{header}\n')

    def finish(self) -> None:
        self._file.close()

    def log_scalar(self, source: Source, value: Scalar) -> None:
        self._columns_values[self.source_to_name(source)] = value
        self._save()

    def log_array(self, source: Source, value: Array) -> None:
        self._columns_values[self.source_to_name(source)] = f"\"{' '.join(str(v) for v in value)}\""
        self._save()

    def log_dict(self, source: Source, value: Dict) -> None:
        self.log_other(source, value)

    def log_other(self, source: Source, value: Any) -> None:
        self._columns_values[self.source_to_name(source)] = f"\"{str(value)}\""
        self._save()

    def _save(self) -> None:
        if len(self._columns_values) == len(self._columns_names):
            line = ','.join(str(self._columns_values[name]) for name in self._columns_names)
            self._file.write(f'{line}\n')
            self._columns_values = {}
