.. _agents_page:

Agents
======

This module is a set of RL agents. You can either choose one of our built-in agents or implement
your agent with the help of :ref:`Custom agents` guide.


BaseAgent
---------

.. currentmodule:: reinforced_lib.agents.base_agent

.. autoclass:: AgentState
    :members:

.. autoclass:: BaseAgent
    :members:


Epsilon-greedy
--------------

.. currentmodule:: reinforced_lib.agents.e_greedy

.. autoclass:: EGreedyState
    :show-inheritance:
    :members:

.. autoclass:: EGreedy
    :show-inheritance:
    :members:


Gradient Bandit
---------------

.. currentmodule:: reinforced_lib.agents.gradient_bandit

.. autoclass:: GradientBanditState
    :show-inheritance:
    :members:

.. autoclass:: GradientBandit
    :show-inheritance:
    :members:


Particle Filter (Core)
----------------------

.. automodule:: reinforced_lib.agents.core.particle_filter
    :show-inheritance:
    :members:


Particle Filter (Wi-Fi)
-----------------------

.. currentmodule:: reinforced_lib.agents.wifi.particle_filter

.. autoclass:: ParticleFilterState
    :show-inheritance:
    :members:

.. autoclass:: ParticleFilter
    :show-inheritance:
    :members:


Thompson Sampling
-----------------

.. currentmodule:: reinforced_lib.agents.thompson_sampling

.. autoclass:: ThompsonSamplingState
    :show-inheritance:
    :members:

.. autoclass:: ThompsonSampling
    :show-inheritance:
    :members:


UCB
---

.. currentmodule:: reinforced_lib.agents.ucb

.. autoclass:: UCBState
    :show-inheritance:
    :members:

.. autoclass:: UCB
    :show-inheritance:
    :members:
