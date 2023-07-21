.. reinforced-lib documentation master file, created by
   sphinx-quickstart on Thu Jun 16 16:58:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. _reinforced-lib:

Welcome to Reinforced-lib's documentation!
==========================================

.. image:: https://img.shields.io/pypi/v/reinforced-lib
    :target: https://pypi.org/project/reinforced-lib/

.. image:: https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg
    :target: https://opensource.org/licenses/MPL-2.0

**Introducing Reinforced-lib:** a lightweight Python library for rapid development of RL solutions. This open-source framework prioritizes ease of use, provides comprehensive documentation, and offers both deep reinforcement learning (DRL) and classic non-neural agents. Built on `JAX <https://jax.readthedocs.io/en/latest/>`_, it facilitates exporting trained models to embedded devices, supporting research, and prototyping with RL algorithms while ensuring high-performance through functional implementation and access to JAX's JIT functionality.

Key components
--------------

Reinforced-lib facilitates seamless interaction between RL agents and the environment. Here are the key components within of the library, represented in the API as different modules.

* **Agents** -- Choose from a variety of RL agents available in the Agents module. These agents are designed to be versatile and work with any environment. If needed, you can even create your own agents using our documented recipes.

* **Extensions** -- Enhance agent observations with domain-specific knowledge by adding a suitable extension from the Extensions module. This step enables seamless agent switching and parameter tuning without extensive reconfiguration.

* **RLib** -- The core module, RLib, provides a simple and intuitive interface to manage agents, use extensions, and configure the logging system. Even if you're not an RL expert, RLib makes it easy to implement the agent-environment interaction loop.

* **Logging** -- The Logging module allows you to monitor agent-environment interactions. Customize and adapt logging to your specific needs, capturing training metrics, internal agent state, or environment status. The library includes various loggers for creating plots and output files, simplifying visualization and data processing.

With a modular design and easy data flow between components, Reinforced-lib ensures quick and effortless replacement or parameter adjustments. The figure below provides a visual representation of Reinforced-lib and the data-flow between its modules.

.. image:: ../resources/data-flow.png
    :width: 450
    :align: center
    :alt: Reinforced-lib architecture and data flow schema

Edge Device Export
------------------

Reinforced-lib is designed to work seamlessly on wireless, low-powered devices, where resources are limited. It's the perfect solution for energy-constrained environments that may struggle with other ML frameworks. You can export your trained models to `TensorFlow Lite <https://www.tensorflow.org/lite>`_ with ease, reducing runtime overhead and optimizing performance through quantization and fixed-point arithmetic. This means you can deploy RL agents on resource-limited devices efficiently.


Table of Contents
-----------------

Explore the power of Reinforced-lib with our easy-to-follow guides and practical examples in the documentation. Unleash the potential of Reinforcement Learning for wireless networks and discover exciting possibilities for your projects. Happy reading!

Guides
~~~~~~

* :ref:`Getting started <getting_started_page>`
* :ref:`Examples <examples_page>`
* :ref:`Custom agents <custom_agents>`
* :ref:`Custom extensions <custom_extensions>`
* :ref:`Custom loggers <custom_loggers>`

API Documentation
~~~~~~~~~~~~~~~~~

* :ref:`API <api_page>`
* :ref:`Agents <agents_page>`
* :ref:`Extensions <extensions_page>`
* :ref:`Logging <logging_page>`
* :ref:`Utils <utils_page>`
* :ref:`Exceptions <exceptions_page>`


.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   Getting started <getting_started>
   Examples <examples>
   Custom agents <custom_agents>
   Custom extensions <custom_extensions>
   Custom loggers <custom_loggers>

.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   :hidden:

   API <api>
   Agents <agents>
   Extensions <extensions>
   Logging <logging>
   Utils <utils>
   Exceptions <exceptions>


Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`search`
