.. _agents:

Agents
======

This module is a set of RL agents. You can either choose one of our built-in agents or implemet
your agent with the help of this short guide.

.. _custom-agents:

Custom agents
-------------

To fully benefit from the reinforced-lib features, including the JAX jit optimization, your agent
should inherit from the :ref:`class <base_agent>` ``BaseAgent``. We will present adding new agent 

.. _base_agent:

BaseAgent
---------

.. currentmodule:: reinforced_lib.agents.base_agent

.. autoclass:: AgentState

.. autoclass:: BaseAgent
    :members:

List of agents
--------------

Thompson Sampling
~~~~~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.agents.thompson_sampling

.. autoclass:: ThompsonSamplingState
    :show-inheritance:

.. autoclass:: ThompsonSampling
    :show-inheritance:

.. autofunction:: init

.. autofunction:: update

.. autofunction:: sample

Particle Filter
~~~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.agents.wifi.particle_filter

.. autoclass:: ParticleFilterState
    :show-inheritance:

.. autoclass:: ParticleFilter
    :show-inheritance:

.. autofunction:: init

.. autofunction:: update

.. autofunction:: sample

