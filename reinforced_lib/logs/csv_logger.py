import json
import os.path
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
from chex import Array, Scalar

from reinforced_lib.logs import BaseLogger, Source
from reinforced_lib.utils import timestamp


class CsvLogger(BaseLogger):
    """
    Logger that saves values in CSV format. It saves the logged values to the CSV file when the experiment is finished.
    ``CsvLogger`` synchronizes the logged values in time. It means that if the same source is logged twice in a row,
    the step number will be incremented for all columns and the logger will move to the next row.

    Parameters
    ----------
    csv_path : str, default="~/rlib-logs-[date]-[time].csv"
        Path to the output file.
    """

    def __init__(self, csv_path: str = None, **kwargs) -> None:
        super().__init__(**kwargs)

        if csv_path is None:
            csv_path = f'rlib-logs-{timestamp()}.csv'
            csv_path = os.path.join(os.path.expanduser("~"), csv_path)

        self._csv_path = csv_path
        self._current_values = set()
        self._step = 0

        self._values = defaultdict(list)
        self._steps = defaultdict(list)

    def finish(self) -> None:
        """
        Saves the logged values to the CSV file.
        """

        file = open(self._csv_path, 'w')
        file.write(','.join(self._values.keys()) + '\n')

        rows, cols = self._step + 1, len(self._values)
        csv_array = np.full((rows, cols), fill_value='', dtype=object)

        for j, (name, values) in enumerate(self._values.items()):
            for i, v in enumerate(values):
                csv_array[self._steps[name][i], j] = v

        for row in csv_array:
            file.write(','.join(map(str, row)) + '\n')

        file.close()

    def log_scalar(self, source: Source, value: Scalar, *_) -> None:
        """
        Logs a scalar as a standard value in a column.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : float
            Scalar to log.
        """

        self._log(source, value)

    def log_array(self, source: Source, value: Array, *_) -> None:
        """
        Logs an array as a JSON string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : array_like
            Array to log.
        """

        if isinstance(value, (np.ndarray, jnp.ndarray)):
            value = value.tolist()

        self._log(source, f"\"{json.dumps(value)}\"")

    def log_dict(self, source: Source, value: dict, *_) -> None:
        """
        Logs a dictionary as a JSON string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : dict
            Dictionary to log.
        """

        self._log(source, f"\"{json.dumps(value)}\"")

    def log_other(self, source: Source, value: any, *_) -> None:
        """
        Logs an object as a JSON string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : any
            Value of any type to log.
        """

        self._log(source, f"\"{json.dumps(value)}\"")

    def _log(self, source: Source, value: any) -> None:
        """
        Saves the logged value and controls the current step.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : any
            Value to log.
        """

        name = self.source_to_name(source)

        if name in self._current_values:
            self._step += 1
            self._current_values.clear()

        self._current_values.add(name)
        self._values[name].append(value)
        self._steps[name].append(self._step)
