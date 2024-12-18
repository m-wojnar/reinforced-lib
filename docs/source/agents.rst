.. _agents_page:

Agents
======

This module is a set of RL agents. You can either choose one of our built-in agents or implement
your agent with the help of the :ref:`Custom agents` guide.


BaseAgent
---------

.. currentmodule:: reinforced_lib.agents.base_agent

.. autoclass:: AgentState
    :members:

.. autoclass:: BaseAgent
    :members:


Deep Q-Learning (DQN)
---------------------

.. currentmodule:: reinforced_lib.agents.deep.dqn

.. autoclass:: DQNState
    :show-inheritance:
    :members:

.. autoclass:: DQN
    :show-inheritance:
    :members:


Double Deep Q-Learning (DDQN)
-----------------------------

.. currentmodule:: reinforced_lib.agents.deep.ddqn

.. autoclass:: DDQNState
    :show-inheritance:
    :members:

.. autoclass:: DDQN
    :show-inheritance:
    :members:


Deep Expected SARSA
-------------------

.. currentmodule:: reinforced_lib.agents.deep.expected_sarsa

.. autoclass:: ExpectedSarsaState
    :show-inheritance:
    :members:

.. autoclass:: ExpectedSarsa
    :show-inheritance:
    :members:


Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------

.. currentmodule:: reinforced_lib.agents.deep.ddpg

.. autoclass:: DDPGState
    :show-inheritance:
    :members:

.. autoclass:: DDPG
    :show-inheritance:
    :members:


Epsilon-greedy
--------------

.. currentmodule:: reinforced_lib.agents.mab.e_greedy

.. autoclass:: EGreedyState
    :show-inheritance:
    :members:

.. autoclass:: EGreedy
    :show-inheritance:
    :members:


Exp3
----

.. currentmodule:: reinforced_lib.agents.mab.exp3

.. autoclass:: Exp3State
    :show-inheritance:
    :members:

.. autoclass:: Exp3
    :show-inheritance:
    :members:


Softmax
-------

.. currentmodule:: reinforced_lib.agents.mab.softmax

.. autoclass:: SoftmaxState
    :show-inheritance:
    :members:

.. autoclass:: Softmax
    :show-inheritance:
    :members:


Thompson sampling
-----------------

.. currentmodule:: reinforced_lib.agents.mab.thompson_sampling

.. autoclass:: ThompsonSamplingState
    :show-inheritance:
    :members:

.. autoclass:: ThompsonSampling
    :show-inheritance:
    :members:


Normal Thompson sampling
------------------------

.. currentmodule:: reinforced_lib.agents.mab.normal_thompson_sampling

.. autoclass:: NormalThompsonSamplingState
    :show-inheritance:
    :members:

.. autoclass:: NormalThompsonSampling
    :show-inheritance:
    :members:


Log-normal Thompson sampling
----------------------------

.. currentmodule:: reinforced_lib.agents.mab.lognormal_thompson_sampling

.. autoclass:: LogNormalThompsonSampling
    :show-inheritance:
    :members:


Upper confidence bound (UCB)
----------------------------

.. currentmodule:: reinforced_lib.agents.mab.ucb

.. autoclass:: UCBState
    :show-inheritance:
    :members:

.. autoclass:: UCB
    :show-inheritance:
    :members:


Random scheduler
----------------

.. currentmodule:: reinforced_lib.agents.mab.scheduler.random

.. autoclass:: RandomSchedulerState
    :show-inheritance:
    :members:

.. autoclass:: RandomScheduler
    :show-inheritance:
    :members:


Round-robin scheduler
---------------------

.. currentmodule:: reinforced_lib.agents.mab.scheduler.round_robin

.. autoclass:: RoundRobinSchedulerState
    :show-inheritance:
    :members:

.. autoclass:: RoundRobinScheduler
    :show-inheritance:
    :members:
