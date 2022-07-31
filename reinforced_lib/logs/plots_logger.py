import os.path
from collections import defaultdict
from functools import partial
from typing import List, Set, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Array, Scalar

from reinforced_lib.logs import BaseLogger
from reinforced_lib.logs.logs_observer import Source


class PlotsLogger(BaseLogger):
    def __init__(self, plots_dir: str = '.', plots_ext: str = 'svg', plots_smoothing: Scalar = 0.6, **kwargs) -> None:
        super().__init__(**kwargs)

        self._plots_dir = plots_dir
        self._plots_ext = plots_ext
        self._plots_smoothing = plots_smoothing

        self._plots_values = defaultdict(list)
        self._plots_names = {}

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
        self._plots_names = dict(map(partial(source_to_column, duplicate_names), sources))

    def finish(self) -> None:
        def lineplot(values: List, alpha: Scalar = 1.0, label: bool = False) -> None:
            if jnp.isscalar(values[0]):
                plt.plot(values, alpha=alpha, c='C0')
            else:
                for i, val in enumerate(jnp.array(values).T):
                    plt.plot(val, alpha=alpha, c=f'C{i % 10}', label=i if label else '')

        for name, values in self._plots_values.items():
            if 1 > self._plots_smoothing > 0:
                smoothed = self._exponential_moving_average(values, self._plots_smoothing)
                lineplot(values, alpha=0.3)
                lineplot(smoothed, label=True)
            else:
                lineplot(values)

            plt.title(name)
            plt.xlabel('step')
            plt.legend()
            plt.savefig(os.path.join(self._plots_dir, f'{name}.{self._plots_ext}'))
            plt.show()

    @staticmethod
    def _exponential_moving_average(values: List, weight: Scalar) -> List:
        smoothed = [values[0]]

        for value in values[1:]:
            smoothed.append((1 - weight) * value + weight * smoothed[-1])

        return smoothed

    def log_scalar(self, source: Source, value: Scalar) -> None:
        name = self._plots_names[source]
        self._plots_values[name].append(value)

    def log_array(self, source: Source, value: Array) -> None:
        name = self._plots_names[source]
        self._plots_values[name].append(value)
