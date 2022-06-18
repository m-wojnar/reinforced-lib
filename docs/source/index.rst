.. reinforced-lib documentation master file, created by
   sphinx-quickstart on Thu Jun 16 16:58:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _reinforced-lib:

Welcome to reinforced-lib's documentation!
==========================================

**Reinforced-lib** is a Python library designed to support research and prototyping using Reinforced Learning
(RL) algorithms. The library can serve as a simple solution with ready to use RL workflows, as well as
an expandable framework with programmable behaviour. Thanks to a fuctional implementation of librarys core,
we can provide full acces to JAX's jit functionality, which boosts the agents performance significantly.

.. code-block:: python

   import reinforced_lib as rfl

   # TODO add some simple example presenting the beuty of our reinforced-lib
   rl = rfl.RLib(
      agent_type=SomeAgent
      env_type=SomeEnv
   )

Integrated 802.11ax support
---------------------------

Library design is distinctly influenced by the desire to support research in Wi-Fi. It can be a tool for
researchers to optimize the Wi-Fi protocols with built-in RL algorithms and provided 802.11ax environment
extension.

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   Getting started <getting_started>
   API module <api_module>
   Environments module <env_module>
   Agents module <agents_module>
   Logging module <log_module>


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
