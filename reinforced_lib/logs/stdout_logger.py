import json
from typing import Any, Dict, List

import jax.numpy as jnp
import numpy as np
from chex import Array, Scalar

from reinforced_lib.logs import BaseLogger, Source


class StdoutLogger(BaseLogger):
    """
    Logger that writes values to the standard output.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._values = {}
        self._names = []

    def init(self, sources: List[Source]) -> None:
        """
        Creates a list of all source names.

        Parameters
        ----------
        sources : list[Source]
            List containing all sources to log.
        """

        self._names = list(map(self.source_to_name, filter(lambda s: s is not None, sources)))

    def finish(self) -> None:
        """
        Prints the last row if there are any unprinted values left.
        """

        if len(self._values) > 0:
            print('\t'.join(f'{name}: {self._values[name]}' for name in self._names if name in self._values))

    def log_scalar(self, source: Source, value: Scalar, custom: bool) -> None:
        """
        Logs a scalar as the standard value.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : float
            Scalar to log.
        custom : bool
            Whether the source is a custom source.
        """

        self._print(source, value, custom)

    def log_array(self, source: Source, value: Array, custom: bool) -> None:
        """
        Logs an array as a JSON [2]_ string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : array_like
            Array to log.
        custom : bool
            Whether the source is a custom source.
        """

        if isinstance(value, (np.ndarray, jnp.ndarray)):
            value = value.tolist()

        self.log_other(source, value, custom)

    def log_dict(self, source: Source, value: Dict, custom: bool) -> None:
        """
        Logs a dictionary as a JSON [2]_ string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : dict
            Dictionary to log.
        custom : bool
            Whether the source is a custom source.
        """

        self.log_other(source, value, custom)

    def log_other(self, source: Source, value: Any, custom: bool) -> None:
        """
        Logs an object as a JSON [2]_ string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : any
            Value of any type to log.
        custom : bool
            Whether the source is a custom source.
        """

        self._print(source, json.dumps(value), custom)

    def _print(self, source: Source, value: Any, custom: bool) -> None:
        """
        Prints a new row to the standard output if all values has already been filled.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : any
            Value of any type to log.
        custom : bool
            Whether the source is a custom source.
        """

        if not custom:
            self._values[self.source_to_name(source)] = value

            if len(self._values) == len(self._names):
                print('\t'.join(f'{name}: {self._values[name]}' for name in self._names))
                self._values = {}
        else:
            print(f'{self.source_to_name(source)}: {value}')
