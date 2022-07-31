import os.path
from collections import defaultdict
from typing import List

import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Array, Scalar

from reinforced_lib.logs import BaseLogger, Source


class PlotsLogger(BaseLogger):
    def __init__(self, plots_dir: str = '.', plots_ext: str = 'svg', plots_smoothing: Scalar = 0.6, **kwargs) -> None:
        super().__init__(**kwargs)

        self._plots_dir = plots_dir
        self._plots_ext = plots_ext
        self._plots_smoothing = plots_smoothing

        assert 1 > self._plots_smoothing >= 0

        self._plots_values = defaultdict(list)
        self._plots_names = []

    def init(self, sources: List[Source]) -> None:
        self._plots_names = list(map(self.source_to_name, sources))

    def finish(self) -> None:
        def lineplot(values: List, alpha: Scalar = 1.0, label: bool = False) -> None:
            if jnp.isscalar(values[0]):
                plt.plot(values, alpha=alpha, c='C0')
            else:
                for i, val in enumerate(jnp.array(values).T):
                    plt.plot(val, alpha=alpha, c=f'C{i % 10}', label=i if label else '')
                plt.legend()

        for name, values in self._plots_values.items():
            smoothed = self._exponential_moving_average(values, self._plots_smoothing)
            lineplot(values, alpha=0.3)
            lineplot(smoothed, label=True)
            plt.title(name)
            plt.xlabel('step')
            plt.savefig(os.path.join(self._plots_dir, f'{name}.{self._plots_ext}'))
            plt.show()

    @staticmethod
    def _exponential_moving_average(values: List, weight: Scalar) -> List:
        smoothed = [values[0]]

        for value in values[1:]:
            smoothed.append((1 - weight) * value + weight * smoothed[-1])

        return smoothed

    def log_scalar(self, source: Source, value: Scalar) -> None:
        self._plots_values[self.source_to_name(source)].append(value)

    def log_array(self, source: Source, value: Array) -> None:
        self._plots_values[self.source_to_name(source)].append(value)
