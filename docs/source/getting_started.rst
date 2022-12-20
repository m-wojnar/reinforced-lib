.. _getting_started_page:

Getting started
===============

Installation
------------

With pip
~~~~~~~~

You can install the latest released version of Reinforced-lib from PyPI via:

.. code-block:: bash

    pip install reinforced-lib

From source 
~~~~~~~~~~~

You can clone source code from our repository:

.. code-block:: bash

    git clone git@github.com:m-wojnar/reinforced-lib.git

And install it with the pip:

.. code-block:: bash

    cd reinforced-lib
    pip install .

You can also install the development dependencies if you want to build the documentation locally:

.. code-block:: bash

    cd reinforced-lib
    pip install ".[dev]"


Basic usage
-----------

The vital benefit of using :ref:`Reinforced-lib <reinforced-lib>` is the simplification of RL training loop. Thanks to
the :ref:`class <RLib Class>` ``Rlib``, we do not need to worry about the agent initialization nor passing the environment
observations to the agent. The library will take care of these tedious tasks. Below we present the basic training loop with
all the simplifications provided by :ref:`Reinforced-lib <reinforced-lib>`.

.. code-block:: python

    from reinforced_lib import RLib
    from reinforced_lib.agents import ThompsonSampling
    from reinforced_lib.exts import IEEE_802_11_ax

    rlib = RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax
    )

After the necessary imports, we create an instance of :ref:`class <RLib Class>` ``Rlib``. We provide the chosen agent type and
the extension appropriate for the problem. This extension will help the agent to infer necessary information from the
environment. You can learn more about extensions in the :ref:`Custom extensions <custom_extensions>` section.

.. code-block:: python

    import gym

    env = gym.make('WifiSimulator-v1')
    env_state = env.reset()

Next, we import OpenAI gym, make an environment, and reset it to an initial state.

.. code-block:: python

    terminated = False
    while not terminated:
        action = rlib.sample(**env_state)
        env_state, reward, done, info = env.step(action)

We can now define the training loop. The boolen ``terminated`` flag controls when to stop taching the agent.
Inside the loop, we call the ``rlib.sample()`` method which passes environment observations to the agent, updates agent's
internal state and returns an action proposed by the agent's policy. We apply the returned action in the environment to get
it's altered state, reward, information about whether this state is terminal and some additional info.

Logging
-------

The logging module provides a simple and powerful API for visualizing and analyzing running algorithm or watching
the training process. You can monitor observations passed to the agent, the agents state, and basic metrics in
real time. Below is the simplest example of using the built-in logger ``StdoutLogger``:

.. code-block:: python

    rlib = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax,
        logger_type=StdoutLogger,
        loggers_sources='n_successful'
    )

You can easily change the logger type, add more sources, and customize parameters of the logger:

.. code-block:: python

    rlib = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax,
        logger_type=PlotsLogger,
        loggers_sources=['n_successful', 'alpha', ('action', SourceType.METRIC)],
        loggers_params={'plots_smoothing': 0.9}
    )

Note that ``n_successful`` is the observation name, ``alpha`` is name of the attribute of the ``ThompsonSampling``
agent, and ``action`` is the name of the metric. You can mix sources names as long as it does not lead to the
inconclusiveness. In the example above, it can be seen that ``action`` is both name of the observation and the metric.
In this case you have to write source name as the tuple containing name and type of the source ``(str, SourceType)``
as in the code above.

You can also plug multiple loggers to one source:

.. code-block:: python

    rlib = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax,
        logger_type=[StdoutLogger, CsvLogger, PlotsLogger],
        loggers_sources='n_successful'
    )

Or mix different loggers and sources:

.. code-block:: python

    rlib = rfl.RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax,
        logger_type=[StdoutLogger, CsvLogger, PlotsLogger],
        loggers_sources=['n_successful', 'alpha', ('action', SourceType.METRIC)]
    )

In this case remember to provide a list of loggers that is the same length as a list of sources, because given loggers
will be used to log values for consecutive sources.


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

Another feature of this saving mechanism is that it allows us to dynamically change training parameters.
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
parameters alteration, but you can modify extension parameters in the same way. You can even control the
the loggers behaviour with the flag ``restore_loggers`` (more on loggers in the :ref:`Logging module <Logging module>`
section).
