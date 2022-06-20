Getting started
===============

.. _installation:

Installation
------------

TODO add section, when pip configured

Basic usage
-----------

The vital interface of  :ref:`reinforced-lib <reinforced-lib>` is the :ref:`class <rlib>` ``Rlib``,
which abstracts the agent-environment interaction. In basic use case, you only need to provide
appropiate agent with the environmet related to your problem domain and the lib will take care of the rest.

.. code-block:: python

    import reinforced_lib as rfl
    from reinforced_lib.agents import Qlearning
    from reinforced_lib.exts import GymExt

    import gym

    rlib = rfl.RLib(
        agent_type=Qlearning
        ext_type=GymExt
    )

    env = gym.make('CartPole-v1')

    state = env.reset()
    done = False
    while not done:
        action = rlib.sample(*state)
        state, reward, done, info = env.step(action)




Advanced Concepts
-----------------

.. note::

    * If you are looking for adding custom **extension** than check out :ref:`custom_exts` section!
    * If you are looking for adding custom **agent** than check out :ref:`custom_agents` section!

Modular architecture
~~~~~~~~~~~~~~~~~~~~

The whole library has a modular architecture, which enables you to  

.. image:: ../resources/reinforced-lib.jpg
    :width: 500
    :alt: reinforced-lib architecture schema

Extending the environment
~~~~~~~~~~~~~~~~~~~~~~~~~


