.. _agents:

Agents
======

This module is a set of RL agents. You can either choose one of our built-in agents or implemet
your agent with the help of this short guide.

.. _custom_agents:

Custom agents
-------------

To fully benefit from the reinforced-lib features, including the JAX jit optimization, your agent
should inherit from the :ref:`abstract class <base_agent>` ``BaseAgent``. We will present adding new agent 

.. _base_agent:

BaseAgent
---------

.. currentmodule:: reinforced_lib.agents.base_agent

.. autoclass:: AgentState
    :members:

.. autoclass:: BaseAgent
    :members:

List of agents
--------------

Thompson Sampling
~~~~~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.agents.thompson_sampling

.. autoclass:: ThompsonSamplingState
    :show-inheritance:
    :members:

.. autoclass:: ThompsonSampling
    :show-inheritance:
    :members:

Particle Filter
~~~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.agents.wifi.particle_filter

.. autoclass:: ParticleFilterState
    :show-inheritance:
    :members:

.. autoclass:: ParticleFilter
    :show-inheritance:
    :members:

Epsilon-greedy
~~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.agents.e_greedy

.. autoclass:: EGreedyState
    :show-inheritance:
    :members:

.. autoclass:: EGreedy
    :show-inheritance:
    :members:
