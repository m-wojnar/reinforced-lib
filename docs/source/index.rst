.. reinforced-lib documentation master file, created by
   sphinx-quickstart on Thu Jun 16 16:58:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


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


.. _reinforced-lib:

Welcome to Reinforced-lib's documentation!
==========================================

**Reinforced-lib** is a Python library designed to support research and prototyping using Reinforced Learning
(RL) algorithms. The library can serve as a simple solution with ready to use RL workflows, as well as
an expandable framework with programmable behaviour. Thanks to a functional implementation of library's core,
we can provide full access to JAX's jit functionality, which boosts the agents performance significantly.

.. code-block:: python

   import gym

   import reinforced_lib as rfl
   from reinforced_lib.agents import ThompsonSampling
   from reinforced_lib.exts import IEEE_802_11_ax

   rlib = rfl.RLib(
       agent_type=ThompsonSampling,
       ext_type=IEEE_802_11_ax
   )

   env = gym.make('WifiSimulator-v1')

   state = env.reset()
   done = False

   while not done:
       action = rlib.sample(**state)
       state, reward, done, info = env.step(action)


Integrated IEEE 802.11ax support
--------------------------------

Library design is distinctly influenced by the desire to support research in Wi-Fi. It can be a tool for
researchers to optimize the Wi-Fi protocols with built-in RL algorithms and provided IEEE 802.11ax environment
extension.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
