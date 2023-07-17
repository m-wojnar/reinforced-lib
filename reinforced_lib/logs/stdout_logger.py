import json

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

    def finish(self) -> None:
        """
        Prints the last row if there are any unprinted values left.
        """

        if len(self._values) > 0:
            print('\t'.join(f'{n}: {v}' for n, v in self._values.items()))

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

        self._log(source, value, custom)

    def log_array(self, source: Source, value: Array, custom: bool) -> None:
        """
        Logs an array as a JSON string.

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

        self._log(source, json.dumps(value), custom)

    def log_dict(self, source: Source, value: dict, custom: bool) -> None:
        """
        Logs a dictionary as a JSON string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : dict
            Dictionary to log.
        custom : bool
            Whether the source is a custom source.
        """

        self._log(source, json.dumps(value), custom)

    def log_other(self, source: Source, value: any, custom: bool) -> None:
        """
        Logs an object as a JSON string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : any
            Value of any type to log.
        custom : bool
            Whether the source is a custom source.
        """

        self._log(source, json.dumps(value), custom)

    def _log(self, source: Source, value: any, custom: bool) -> None:
        """
        Prints a new row to the standard output if there is a new value for a
        standard source or the source is custom.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : any
            Value of any type to log.
        custom : bool
            Whether the source is a custom source.
        """

        name = self.source_to_name(source)

        if not custom:
            if name in self._values:
                print('\t'.join(f'{n}: {v}' for n, v in self._values.items()))
                self._values = {}

            self._values[name] = value
        else:
            print(f'{name}: {value}')
