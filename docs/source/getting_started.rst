.. _getting_started_page:

Getting started
===============

Installation
------------

With pip
~~~~~~~~

You can install the latest version of Reinforced-lib from PyPI via:

.. code-block:: bash

    pip install reinforced-lib

From source 
~~~~~~~~~~~

You can clone the source code from our repository

.. code-block:: bash

    git clone git@github.com:m-wojnar/reinforced-lib.git

and install it with the pip:

.. code-block:: bash

    cd reinforced-lib
    pip install .

You can also install the development dependencies if you want to build the documentation locally:

.. code-block:: bash

    cd reinforced-lib
    pip install ".[dev]"


Basic usage
-----------

The vital benefit of using :ref:`Reinforced-lib <reinforced-lib>` is a simplification of the RL training loop. Thanks to
the :ref:`class <RLib Class>` ``Rlib``, we do not need to worry about the agent initialization nor passing the environment
observations to the agent. The library takes care of these tedious tasks. Below, we present the basic training loop with
all the simplifications provided by :ref:`Reinforced-lib <reinforced-lib>`.

.. code-block:: python

    from reinforced_lib import RLib
    from reinforced_lib.agents.mab import ThompsonSampling
    from reinforced_lib.exts.wifi import IEEE_802_11_ax_RA

    rlib = RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax_RA
    )

After the necessary imports, we create an instance of :ref:`class <RLib Class>` ``Rlib``. We provide the chosen agent type and
the appropriate extension for the problem. This extension will help the agent to infer necessary information from the
environment. To learn more about extensions check out the :ref:`Custom extensions <custom_extensions>` section.

.. code-block:: python

    import gymnasium as gym

    env = gym.make('WifiSimulator-v1')
    env_state = env.reset()

Next, we import Gymnasium, make an environment, and reset it to an initial state.

.. code-block:: python

    terminated = False
    while not terminated:
        action = rlib.sample(**env_state)
        env_state, reward, terminal, truncated, info = env.step(action)

We can now define the training loop. The boolean ``terminal`` and ``truncated`` flags control when to stop training the agent.
Inside the loop, we call the ``rlib.sample()`` method which passes the environment observations to the agent, updates
the agent's internal state and returns an action proposed by the agent's policy. We apply the returned action in the
environment to get its altered state, reward, information whether this state is terminal, and some additional info.

Logging
-------

The logging module provides a simple yet powerful API for visualizing and analyzing the running algorithm or watching
the training process. You can monitor any observations passed to the agent, the agent's state, and the basic metrics in
real time. Below is the simplest example of using the built-in logger ``StdoutLogger``:

.. code-block:: python

    rlib = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax_RA,
        logger_types=StdoutLogger,
        logger_sources='n_successful'
    )

You can easily change the logger type, add more sources, and customize the parameters of the logger:

.. code-block:: python

    rlib = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax_RA,
        logger_types=PlotsLogger,
        logger_sources=['n_successful', 'alpha', ('action', SourceType.METRIC)],
        logger_params={'plots_smoothing': 0.9}
    )

Note that ``n_successful`` is the observation name, ``alpha`` is name of the attribute of the ``ThompsonSampling``
agent, and ``action`` is the name of the metric. You can mix sources names as long as it does not lead to
inconclusiveness. In the example above, it can be seen that ``action`` is both the name of the observation and the metric.
In this case, you have to write the source name as a tuple containing a name and the type of the source ``(str, SourceType)``
as in the code above.

You can also plug multiple loggers to one source:

.. code-block:: python

    rlib = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax_RA,
        logger_types=[StdoutLogger, CsvLogger, PlotsLogger],
        logger_sources='n_successful'
    )

Or mix different loggers and sources:

.. code-block:: python

    rlib = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax_RA,
        logger_types=[StdoutLogger, CsvLogger, PlotsLogger],
        logger_sources=['n_successful', 'alpha', ('action', SourceType.METRIC)]
    )

In this case remember to provide a list of loggers that has the same length as the list of sources, because given loggers
will be used to log values for consecutive sources.


Saving experiments
------------------

The ``RLib`` :ref:`class <RLib Class>` provides an API for saving your experiment in a compressed ``.lz4`` format.
You can later reconstruct the experiment state and continue from the exact point where you ended or you can
alter some training parameters during the reloading process.


Full reconstruction
~~~~~~~~~~~~~~~~~~~

We can imagine a scenario where we set up the experiment, perform a little training, and then we need to take a break.
Therefore, we save the experiment at the latest state that we would later want to carry on from. When we are ready to continue
with the training, we load the whole experiment to a new RLib instance.

.. code-block:: python

    import reinforced_lib as rfl

    from reinforced_lib.agents.mab import ThompsonSampling
    from reinforced_lib.exts.wifi import IEEE_802_11_ax_RA
    
    # Setting up the experiment
    rl = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax_RA
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


Dynamic parameter change
~~~~~~~~~~~~~~~~~~~~~~~~~

Another feature of the saving mechanism is that it allows us to dynamically change training parameters.
Let us recall the above example and modify it a little. We now want to modify on-the-fly the ``decay``
parameter (responsible for the 'memory' of the Thompson sampling agent).

.. code-block:: python

    import reinforced_lib as rfl

    from reinforced_lib.agents.mab import ThompsonSampling
    from reinforced_lib.exts.wifi import IEEE_802_11_ax_RA
    
    # Setting up the experiment
    rl = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax_RA
    )

    # Do some training
    # ...

    # Saving experiment state for later
    rl.save("<checkpoint-path>")

    # Load the saved training with altered parameters
    rl = RLib.load("<checkpoint-path>", agent_params={"decay": new_decay})

    # Continue the training with new parameters
    # ...

We can change as many parameters we want. The provided example is constrained only to the agent's
parameter alteration, but you can modify the extension's parameters in the same way. You can even control the
the logger's behaviour with the flag ``restore_loggers`` (more on loggers in the :ref:`Logging module <The logging module>`
section).
