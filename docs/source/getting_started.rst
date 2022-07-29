Getting started
===============

Installation
------------

TODO add section, when pip configured

Basic usage
-----------

The vital interface of  :ref:`reinforced-lib <reinforced-lib>` is the :ref:`class <RLib Class>` ``Rlib``,
which abstracts the agent-environment interaction. In basic use case, you only need to provide
appropiate agent with the environmet related to your problem domain and the lib will take care of the rest.

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


Modular architecture
--------------------

The whole library has a modular architecture, which enables you to  

.. image:: ../resources/reinforced-lib.jpg
    :width: 500
    :alt: reinforced-lib architecture schema
