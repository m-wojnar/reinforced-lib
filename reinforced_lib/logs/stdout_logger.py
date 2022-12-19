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
        Creates a list of all sources names.

        Parameters
        ----------
        sources : list[Source]
            List containing all sources to log.
        """

        self._names = list(map(self.source_to_name, sources))

    def finish(self) -> None:
        """
        Prints the last row if there are any unprinted values left.
        """

        if len(self._values) > 0:
            for name in self._names:
                if name in self._values:
                    print(f'{name}: {self._values[name]}', end='\t')

            print()

    def log_scalar(self, source: Source, value: Scalar) -> None:
        """
        Logs a scalar as the standard value.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : float
            Scalar to log.
        """

        self._print(source, value)

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
        """

        self._print(source, json.dumps(value))

    def _print(self, source: Source, value: Any) -> None:
        """
        Prints a new row to the standard output if there is any value update.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : any
            Value of any type to log.
        """

        if not (name := self.source_to_name(source)) in self._values:
            self._values[name] = value
            return

        for name in self._names:
            if name in self._values:
                print(f'{name}: {self._values[name]}', end='\t')

        self._values = {}
        print()
