Custom loggers
==============

Loggers are a very helpful tool to visualize and analyze running algorithm or watch the training process. You can
monitor observations passed to the agent, the agents state, and basic metrics in real time.


Customizing loggers
-------------------

To create your own extension, you should inherit from the :ref:`abstract class <BaseLogger>` ``BaseLogger``.
We will present adding custom logger on the example of the ``CsvLogger`` :ref:`logger <CsvLogger>` .

.. code-block:: python

    class CsvLogger(BaseLogger)

Firstly, we must define a loggers constructor. ``__init__`` function can take any arguments that can be later passed
by the ``loggers_params`` parameter in the ``RLib`` :ref:`class <RLib class>` constructor, but remember to always
include the ``**kwargs`` in the arguments list. Values provided in ``loggers_params`` are passed to instances
of all loggers listed in the ``loggers_type`` parameter, so it is very important to choose parameter names carefully.
For example constructor parameters of the ``PlotsLogger`` start with prefix ``plots_*`` and parameters of the
``CsvLogger`` start with ``csv_*``. Below is an example constructor of the ``CsvLogger``:

.. code-block:: python

    def __init__(self, csv_path: str = None, **kwargs) -> None:
        super().__init__(**kwargs)

        if csv_path is None:
            now = datetime.now()
            csv_path = f'rlib-logs-{now.strftime("%Y%m%d")}-{now.strftime("%H%M%S")}.csv'
            csv_path = os.path.join(os.path.expanduser("~"), csv_path)

        self._file = open(csv_path, 'w')

        self._columns_values = {}
        self._columns_names = []

``BaseLogger`` :ref:`class <BaseLogger>` offers overwriting the ``init`` method which can be used to initialize
logger attributes given the list of all sources. For example the ``CsvLogger`` creates list of all sources names
and writes header to the output file in that method:

.. code-block:: python

    def init(self, sources: List[Source]) -> None:
        self._columns_names = list(map(self.source_to_name, sources))
        header = ','.join(self._columns_names)
        self._file.write(f'{header}\n')

There is one more useful method that can be used to finalize loggers work - the ``finish`` method. It is a
recommended way to save data, close opened files, show generated plots or perform some clean up. The ``finish``
method is called automatically when instance of the ``RLib`` class is deleted. You can also force the finalization
by calling ``rl.finish()``.

.. code-block:: python

    def finish(self) -> None:
        self._file.close()

Now we can move on to the core methods of the custom logger, which are ``log_scalar``, ``log_array``, ``log_dict``,
and ``log_other``. These methods are used to log scalar values, arrays, dictionaries and other objects accordingly.
The ``LogsObserver`` :ref:`class <LogsObserver>` is responsible for passing logged values to appropriate methods.
We have to overwrite mentioned methods to allow our logger to log values of a given type. Fox example, let's look
at the ``log_scalar`` and the ``log_other`` methods of the ``CsvLogger``:

.. code-block:: python

    def log_scalar(self, source: Source, value: Scalar) -> None:
        self._columns_values[self.source_to_name(source)] = value
        self._save()

.. code-block:: python

    def log_other(self, source: Source, value: Any) -> None:
        self._columns_values[self.source_to_name(source)] = f"\"{json.dumps(value)}\""
        self._save()

These are very simple methods that logs scalars and values of other types. The ``log_scalar`` function just takes
raw scalar and saves it with protected method ``_save`` of the ``CsvLogger``. Similarly, the ``log_other`` function
converts a given value to the JSON format and then calls ``_save``. Note that both methods use ``source_to_name``
of the ``BaseLogger`` that converts source to the string. If the source is a string (just a name of an observation,
state or metric), the method returns that string. Otherwise if the source is a tuple ``(str, SourceType)``,
the function returns string ``"[name]-[source type name]"``.

If the logger is not able to log a value of some type (for example it could be hard to plot a dictionary or a custom
object), we do not have to implement corresponding ``log_*`` method. If the user will try to log a value of that
type with this logger, it will raise the ``UnsupportedLogTypeError`` :ref:`exception <Exceptions>`.


Template logger
---------------

Here is the above code in one piece. You can copy-paste it and use as an inspiration to create your own logger.
Full source code of the ``CsvLogger`` can be found `here <https://github.com/m-wojnar/reinforced-lib/blob/main/reinforced_lib/logs/csv_logger.py>`_.

.. code-block:: python

    import json
    import os.path
    from datetime import datetime
    from typing import Any, Dict, List

    import jax.numpy as jnp
    import numpy as np
    from chex import Array, Scalar

    from reinforced_lib.logs import BaseLogger, Source


    class CsvLogger(BaseLogger):
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
            self._columns_names = list(map(self.source_to_name, sources))
            header = ','.join(self._columns_names)
            self._file.write(f'{header}\n')

        def finish(self) -> None:
            self._file.close()

        def log_scalar(self, source: Source, value: Scalar) -> None:
            self._columns_values[self.source_to_name(source)] = value
            self._save()

        def log_array(self, source: Source, value: Array) -> None:
            if isinstance(value, (np.ndarray, jnp.ndarray)):
                value = value.tolist()

            self.log_other(source, value)

        def log_dict(self, source: Source, value: Dict) -> None:
            self.log_other(source, value)

        def log_other(self, source: Source, value: Any) -> None:
            self._columns_values[self.source_to_name(source)] = f"\"{json.dumps(value)}\""
            self._save()

        def _save(self) -> None:
            if len(self._columns_values) == len(self._columns_names):
                line = ','.join(str(self._columns_values[name]) for name in self._columns_names)
                self._file.write(f'{line}\n')
                self._columns_values = {}
