import json
import os.path
from datetime import datetime
from typing import Any, Dict, List

import jax.numpy as jnp
import numpy as np
from chex import Array, Scalar

from reinforced_lib.logs import BaseLogger, Source


class CsvLogger(BaseLogger):
    """
    Logger that saves values in CSV [1]_ format.

    Parameters
    ----------
    csv_path : str, default="~/rlib-logs-[date]-[time].csv"
        Path to the output file.

    References
    ----------
    .. [1] Shafranovich, Y. (2005). Common Format and MIME Type for Comma-Separated Values (CSV) Files
       (RFC No. 4180). RFC Editor. https://www.rfc-editor.org/rfc/rfc4180.txt
    """

    def __init__(self, csv_path: str = None, **kwargs) -> None:
        super().__init__(**kwargs)

        if csv_path is None:
            now = datetime.now()
            csv_path = f'rlib-logs-{now.strftime("%Y%m%d")}-{now.strftime("%H%M%S")}.csv'
            csv_path = os.path.join(os.path.expanduser("~"), csv_path)

        self._file = open(csv_path, 'w')

        self._columns_values = {}
        self._columns_names = []

    def init(self, sources: List[Source]) -> None:
        """
        Creates a list of all source names and writes the header to the output file.

        Parameters
        ----------
        sources : list[Source]
            List containing all sources to log.
        """

        self._columns_names = list(map(self.source_to_name, sources))
        header = ','.join(self._columns_names)
        self._file.write(f'{header}\n')

    def finish(self) -> None:
        """
        Closes the output file.
        """

        self._file.close()

    def log_scalar(self, source: Source, value: Scalar) -> None:
        """
        Logs a scalar as a standard value in a column.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : float
            Scalar to log.
        """

        self._columns_values[self.source_to_name(source)] = value
        self._save()

    def log_array(self, source: Source, value: Array) -> None:
        """
        Logs an array as a JSON [2]_ string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : array_like
            Array to log.
        """

        if isinstance(value, (np.ndarray, jnp.ndarray)):
            value = value.tolist()

        self.log_other(source, value)

    def log_dict(self, source: Source, value: Dict) -> None:
        """
        Logs a dictionary as a JSON [2]_ string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : dict
            Dictionary to log.
        """

        self.log_other(source, value)

    def log_other(self, source: Source, value: Any) -> None:
        """
        Logs an object as a JSON [2]_ string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : any
            Value of any type to log.

        References
        ----------
        .. [2] Pezoa, F., Reutter, J. L., Suarez, F., Ugarte, Mart&#237;n, & Vrgo&#269;c, Domagoj. (2016).
           Foundations of JSON schema. In Proceedings of the 25th International Conference on World Wide Web
           (pp. 263–273).
        """

        self._columns_values[self.source_to_name(source)] = f"\"{json.dumps(value)}\""
        self._save()

    def _save(self) -> None:
        """
        Writes a new row to the output file if the values for all columns has already been filled.
        """

        if len(self._columns_values) == len(self._columns_names):
            line = ','.join(str(self._columns_values[name]) for name in self._columns_names)
            self._file.write(f'{line}\n')
            self._columns_values = {}
