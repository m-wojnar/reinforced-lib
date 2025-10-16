.. _reinforced-lib:

Welcome to Reinforced-lib's documentation!
==========================================

.. image:: https://img.shields.io/pypi/v/reinforced-lib
    :target: https://pypi.org/project/reinforced-lib/

.. image:: https://img.shields.io/badge/Licenses-CC_BY_4.0-brightgreen
    :target: https://github.com/m-wojnar/reinforced-lib/blob/main/LICENSE

.. image:: https://github.com/m-wojnar/reinforced-lib/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/m-wojnar/reinforced-lib/actions

.. image:: https://readthedocs.org/projects/reinforced-lib/badge/?version=latest
    :target: https://reinforced-lib.readthedocs.io/

.. image:: https://img.shields.io/badge/DOI-10.1016/j.softx.2024.101706-blue.svg
    :target: https://doi.org/10.1016/j.softx.2024.101706

.. image:: https://img.shields.io/badge/DOI-10.1016/j.softx.2025.102400-blue.svg
    :target: https://doi.org/10.1016/j.softx.2025.102400

**Introducing Reinforced-lib:** a lightweight Python library for rapid development of reinforcement-learning (RL) solutions. It is open-source, prioritizes ease of use, provides comprehensive documentation, and offers both deep reinforcement learning (DRL) and classic non-neural agents. Built on `JAX <https://jax.readthedocs.io/en/latest/>`_, it facilitates exporting trained models to embedded devices, and makes it great for research and prototyping with RL algorithms. Access to JAX's JIT functionality ensure high-performance results.

Key components
--------------

Reinforced-lib facilitates seamless interaction between RL agents and the environment. Here are the key components within of the library, represented in the API as different modules.

* **RLib** -- The core module which provides a simple and intuitive interface to manage agents, use extensions, and configure the logging system. Even if you're not an RL expert, *RLib* makes it easy to implement the agent-environment interaction loop.

* **Agents** -- Choose from a variety of RL agents available in the *Agents* module. These agents are designed to be versatile and work with any environment. If needed, you can even create your own agents using our documented recipes.

* **Extensions** -- Enhance agent observations with domain-specific knowledge by adding a suitable extension from the *Extensions* module. This module enables seamless agent switching and parameter tuning without extensive reconfiguration.

* **Loggers** -- This module allows you to monitor agent-environment interactions. Customize and adapt logging to your specific needs, capturing training metrics, internal agent state, or environment observations. The library includes various loggers for creating plots and output files, simplifying visualization and data processing.

The figure below provides a visual representation of Reinforced-lib and the data-flow between its modules.

.. image:: ../resources/data-flow.png
    :width: 450
    :align: center
    :alt: Reinforced-lib architecture and data flow schema

JAX Backend
-----------

Our library is built on top of JAX, a high-performance numerical computing library. JAX makes it easy to implement RL algorithms efficiently. It provides powerful transformations, including JIT compilation, automatic differentiation, vectorization, and parallelization. Our library is fully compatible with DeepMind's JAX ecosystem, granting access to state-of-the-art RL models and helper libraries. JIT compilation significantly accelerates execution and ensures portability across different architectures (CPUs, GPUs, TPUs) without requiring code modifications. JAX offers another benefit through its robust pseudorandom number generator system, employed in our library to guarantee result reproducibility. This critical aspect of scientific research is frequently underestimated but remains highly significant.

Edge Device Export
------------------

Reinforced-lib is designed to work seamlessly on wireless, low-powered devices, where resources are limited. It's the perfect solution for energy-constrained environments that may struggle with other ML frameworks. You can export your trained models to `TensorFlow Lite <https://www.tensorflow.org/lite>`_ with ease, reducing runtime overhead and optimizing performance. This means you can deploy RL agents on resource-limited devices efficiently.

Table of Contents
-----------------

Explore the power of Reinforced-lib with our easy-to-follow guides and practical examples in the documentation. Unleash the potential of RL for wireless networks and discover exciting possibilities for your projects. Happy reading!

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
* :ref:`Loggers <loggers_page>`
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
   Loggers <loggers>
   Utils <utils>
   Exceptions <exceptions>


Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`search`


Citing Reinforced-lib
---------------------

To cite this repository, please use the following BibTeX entry for the Reinforced-lib paper:

.. code-block:: latex

    @article{reinforcedlib2022,
      author = {Maksymilian Wojnar and Szymon Szott and Krzysztof Rusek and Wojciech Ciezobka},
      title = {{R}einforced-lib: {R}apid prototyping of reinforcement learning solutions},
      journal = {SoftwareX},
      volume = {26},
      pages = {101706},
      year = {2024},
      issn = {2352-7110},
      doi = {https://doi.org/10.1016/j.softx.2024.101706},
      url = {https://www.sciencedirect.com/science/article/pii/S2352711024000773}
    }

If you are using Reinforced-lib v2.0.0 or later, you can alse cite the following paper:

.. code-block:: latex

    @article{reinforcedlib2025,
      author = {Maksymilian Wojnar},
      title = {{V}ersion v2.0.0 — {R}einforced-lib: {R}apid prototyping of reinforcement learning solutions},
      journal = {SoftwareX},
      volume = {32},
      pages = {102400},
      year = {2025},
      issn = {2352-7110},
      doi = {https://doi.org/10.1016/j.softx.2025.102400},
      url = {https://www.sciencedirect.com/science/article/pii/S2352711025003668},
    }
