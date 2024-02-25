.. _loggers_page:

Loggers module
==============

This module is a set of loggers. You can either choose one of our built-in loggers or implement
your logger with the help of the :ref:`Custom loggers` guide.


BaseLogger
----------

.. currentmodule:: reinforced_lib.logs.base_logger

.. autoclass:: BaseLogger
    :members:


LogsObserver
------------

.. currentmodule:: reinforced_lib.logs.logs_observer

.. autoclass:: LogsObserver
    :members:


CsvLogger
---------

.. currentmodule:: reinforced_lib.logs.csv_logger

.. autoclass:: CsvLogger
    :show-inheritance:
    :members:


StdoutLogger
------------

.. currentmodule:: reinforced_lib.logs.stdout_logger

.. autoclass:: StdoutLogger
    :show-inheritance:
    :members:


PlotsLogger
-----------

.. currentmodule:: reinforced_lib.logs.plots_logger

.. autoclass:: PlotsLogger
    :show-inheritance:
    :members:


TensorboardLogger
-----------------

.. currentmodule:: reinforced_lib.logs.tb_logger

.. autoclass:: TensorboardLogger
    :show-inheritance:
    :members:


WeightsAndBiasesLogger
----------------------

.. currentmodule:: reinforced_lib.logs.wandb_logger

.. autoclass:: WeightsAndBiasesLogger
    :show-inheritance:
    :members:
