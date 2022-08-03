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


Saving experiments
------------------

``RLib`` :ref:`class <RLib Class>` provides an API for saving your experiment in a compressed ``.lz4`` format.
You can later reconstruct the experiment state and continue from the exact point where you have ended or you can
alter some training parameters during the reloading process.


Full reconstruction
~~~~~~~~~~~~~~~~~~~

We can imagine a scenario, where we set up the experiment, perform a little training, and then we need to make a break,
so we save the experiment at some state that we would later want to carry on from. When we are ready to continue with
the training, we can load the whole experiment to a new RLib instance.

.. code-block:: python

    import reinforced_lib as rfl

    from reinforced_lib.agents import ThompsonSampling
    from reinforced_lib.exts import IEEE_802_11_ax
    
    # Setting up the experiment
    rl = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax
    )

    # Do some training
    # ...

    # Saving experiment state for later
    rl.save("<checkpoint-path>")

    # Do some other staff, quit the script if you want.

    # Load the saved training
    rl = RLib.load("<checkpoint-path>")

    # Continue the training
    # ...


Dynamic parameters change
~~~~~~~~~~~~~~~~~~~~~~~~~

Another feature of this saving mechanism is that it allows us to dynamicly change training parameters.
Let's recall the above example and modify it a little. We now want to modify on-the-run the ``decay``
parameter (responsible for the 'memory' of the thompson sampling agent).

.. code-block:: python

    import reinforced_lib as rfl

    from reinforced_lib.agents import ThompsonSampling
    from reinforced_lib.exts import IEEE_802_11_ax
    
    # Setting up the experiment
    rl = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax
    )

    # Do some training
    # ...

    # Saving experiment state for later
    rl.save("<checkpoint-path>")

    # Load the saved training with altered parameters
    rl = RLib.load("<checkpoint-path>", agent_params={"decay": new_decay})

    # Continue the training with new parameters
    # ...

You can change as many parameters we want. The provided example is constrained only to the agent
parameters alteration, but you can modify extension parameters in the same way. You can even controll the
the loggers behaviour with the flag ``restore_loggers`` (more on loggers in the :ref:`Logging module <Logging module>`
section).


Modular architecture
--------------------

The whole library has a modular architecture, which enables you to  

.. image:: ../resources/reinforced-lib.jpg
    :width: 500
    :alt: reinforced-lib architecture schema
