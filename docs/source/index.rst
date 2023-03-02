.. reinforced-lib documentation master file, created by
   sphinx-quickstart on Thu Jun 16 16:58:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. _reinforced-lib:

Welcome to Reinforced-lib's documentation!
==========================================

**Reinforced-lib** is a Python library designed to support research and prototyping using reinforcement learning
(RL) algorithms. The library can serve as a simple solution with ready to use RL workflows as well as
an expandable framework with programmable behaviour. Thanks to the functional implementation of the library's core,
we can provide full access to JAX's JIT functionality, which boosts the agent's performance significantly.

.. code-block:: python

   from reinforced_lib import RLib
   from reinforced_lib.agents.mab import ThompsonSampling
   from reinforced_lib.exts.wifi import IEEE_802_11_ax_RA

   import gymnasium as gym


   rlib = RLib(
      agent_type=ThompsonSampling,
      ext_type=IEEE_802_11_ax_RA
   )

   env = gym.make('WifiSimulator-v1')
   env_state, _ = env.reset()

   terminated = False
   while not terminated:
      action = rlib.sample(**env_state)
      env_state, reward, terminated, *_ = env.step(action)


Integrated IEEE 802.11ax support
--------------------------------

The library's design is highly influenced by the desire to support research in Wi-Fi. It can be a tool for
researchers to optimize Wi-Fi protocols with built-in RL algorithms and provided the IEEE 802.11ax environment
extension.

Modular architecture
--------------------

**Reinforced-lib** can be well characterized by its modular architecture which makes the library flexible, universal,
and easy-to-use. Its key parts are placed in separate modules and connected in a standardized way to provide versatility
and the possibility to extend individual modules in the future. Nevertheless, Reinforced-lib is a single piece of software
that can be easily used, thanks to the topmost module, which ensures a simple and common interface for all agents.

.. image:: ../resources/architecture.jpg
    :width: 500
    :alt: Reinforced-lib architecture schema

The API module
~~~~~~~~~~~~~~

The API module is the top layer of the library; it exposes a simple and intuitive interface that makes the library easy
to use. There are several important methods, one of them is responsible for creating a new agent. Another takes the
observations from the environment as input, updates the state of the agent, and returns the next action proposed by the agent.
The last two methods are used to persist the state of the agent by storing it in memory.

The extensions module
~~~~~~~~~~~~~~~~~~~~~

The extensions module consists of containers with domain-specific knowledge and ensures the proper use of universal agents
implemented in **Reinforced-lib**. If a specific problem is implemented in the form of an extension, the module infers and
provides the appropriate data to the agent, and at the same time requires adequate, corresponding values from the user.

The agents module
~~~~~~~~~~~~~~~~~

The agents module is a collection of universal algorithms, which are called "agents" in the RL community. Each agent has
a similar API to communicate with the Extensions module, which ensures its versatility and expandability. In this release
of **Reinforced-lib**, we focus on the `multi-armed bandit problem <https://en.wikipedia.org/wiki/Multi-armed_bandit>`_,
hence the implemented agents are related to this task.

The logging module
~~~~~~~~~~~~~~~~~~

The logging module is responsible for collecting data from other modules and observing their state in real time.
It also has great potential in using the library to create new RL agents - it can be used to develop, evaluate,
and debug new agents by observing decisions they make; record and visualize how environment state changes in time;
or provide a simple way to obtain a training summary, metrics, and logs.


Table of Contents
-----------------

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
   Exceptions <exceptions>


Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`search`
