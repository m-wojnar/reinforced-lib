from functools import partial
from typing import Any, Dict, List, Set, Tuple

from chex import Array, Scalar

from reinforced_lib.logs import BaseLogger
from reinforced_lib.logs.logs_observer import Source


class CsvLogger(BaseLogger):
    def __init__(self, csv_path: str = 'output.csv', **kwargs) -> None:
        super().__init__(**kwargs)
        self._file = open(csv_path, 'w')

        self._columns_values = {}
        self._columns_names = {}

    def init(self, sources: List[Source]) -> None:
        def find_duplicate_names(sources: List[Source]) -> Set:
            all_names, duplicate_names = set(), set()

            for source in sources:
                source = source[0] if isinstance(source, tuple) else source
                if source in all_names:
                    duplicate_names.add(source)

                all_names.add(source)

            return duplicate_names

        def source_to_column(duplicate_names: Set, source: Source) -> Tuple[Source, str]:
            if isinstance(source, tuple) and source[0] in duplicate_names:
                return source, f'{source[0]}-{source[1].value}'
            else:
                return source, source

        duplicate_names = find_duplicate_names(sources)
        self._columns_names = dict(map(partial(source_to_column, duplicate_names), sources))

        header = ','.join(self._columns_names)
        self._file.write(f'{header}\n')

    def finish(self) -> None:
        self._file.close()

    def log_scalar(self, source: Source, value: Scalar) -> None:
        name = self._columns_names[source]
        self._columns_values[name] = value
        self._save()

    def log_array(self, source: Source, value: Array) -> None:
        name = self._columns_names[source]
        self._columns_values[name] = ' '.join(str(v) for v in value)
        self._save()

    def log_dict(self, source: Source, value: Dict) -> None:
        name = self._columns_names[source]
        self._columns_values[name] = str(value)
        self._save()

    def log_other(self, source: Source, value: Any) -> None:
        if isinstance(value, str):
            name = self._columns_names[source]
            self._columns_values[name] = value
            self._save()

    def _save(self):
        if len(self._columns_values) == len(self._columns_names):
            line = ','.join(str(self._columns_values[name]) for name in self._columns_names.values())
            self._file.write(f'{line}\n')
            self._columns_values = {}
