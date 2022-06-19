Agents
======

This module is a set of RL agents. You can either choose one of our built-in agents or implemet
your agent with the help of this short guide.

.. _custom-agents:

Custom agents
-------------

To fully benefit from the reinforced-lib features, including the JAX jit optimization, your agent
should inherit from the :ref:`class <base_agent>` ``BaseAgent``. We will present adding new agent 

BaseAgent
---------

.. currentmodule:: reinforced_lib.agents.base_agent

.. _base_agent:

.. autoclass:: BaseAgent
    :members:

List of agents
--------------

Thompson Sampling
~~~~~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.agents.thompson_sampling

.. autoclass:: ThompsonSampling
    :show-inheritance:

Particle Filter
~~~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.agents.wifi.particle_filter

.. autoclass:: ParticleFilter
    :show-inheritance:

