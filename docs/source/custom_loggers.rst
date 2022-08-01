Custom loggers
==============

Loggers are a very helpful tool to visualize and analyze running algorithm or training process. You can
monitor observations passed to the agent, the agents state, and basic metrics in real time.


Customizing loggers
-------------------

To create your own extension, you should inherit from the :ref:`abstract class <BaseLogger>` ``BaseLogger``.
We will present adding custom logger on an examples of the ``PlotsLogger`` :ref:`logger <PlotsLogger>`
and the ``CsvLogger`` :ref:`logger <CsvLogger>` .

.. code-block:: python

    class PlotsLogger(BaseLogger)

Firstly, we must define a loggers constructor. ``__init__`` function can take any arguments that can be later passed
by the ``loggers_params`` dictionary in the ``RLib`` :ref:`class <RLib class>`, but remember to always include
the ``**kwargs`` argument in the arguments list (it is required by the internal behaviour of the library).

.. code-block:: python

    def __init__(self, plots_dir: str = '.', plots_ext: str = 'svg', plots_smoothing: Scalar = 0.6, **kwargs) -> None:
        super().__init__(**kwargs)

        self._plots_dir = plots_dir
        self._plots_ext = plots_ext
        self._plots_smoothing = plots_smoothing

        self._plots_values = defaultdict(list)
        self._plots_names = []

``BaseLogger`` :ref:`class <BaseLogger>` offers overwriting the ``init`` method which can be used to initialize
logger attributes given the list of all sources. For example the ``CsvLogger`` creates list of all sources names
and writes header to the output file in that method:

.. code-block:: python

    def init(self, sources: List[Source]) -> None:
        self._columns_names = list(map(self.source_to_name, sources))
        header = ','.join(self._columns_names)
        self._file.write(f'{header}\n')

