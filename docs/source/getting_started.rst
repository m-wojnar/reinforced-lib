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

To have an easy access to the `example files <https://github.com/m-wojnar/reinforced-lib/tree/main/examples>`_,
you can clone the source code from our repository, and than install it locally with pip:

.. code-block:: bash

    git clone git@github.com:m-wojnar/reinforced-lib.git
    cd reinforced-lib
    pip install .

In the spirit of making Reinforced-lib a lightweight solution, we included only the necessary dependencies in the base requirements. To fully benefit from Reinforced-lib conveniences, like TF Lite export, install with the "full" suffix:

.. code-block:: bash

    pip install ".[full]"


.. note::

    Since we tested the Reinforced-lib on Python 3.9, we recommend using Python 3.9+.


Basic usage
-----------

The vital benefit of using Reinforced-lib is a simplification of the agent-environment interaction loop. Thanks to the
``Rlib`` :ref:`class <RLib Class>`, the initialization of the agent and passing the environment state to the agent are
significantly simplified. Below, we present the basic training loop with the simplifications provided by Reinforced-lib.

.. code-block:: python

    import gymnasium as gym
    import haiku as hk
    import optax
    from chex import Array

    from reinforced_lib import RLib
    from reinforced_lib.agents.deep import QLearning
    from reinforced_lib.exts import Gymnasium


    @hk.transform_with_state
    def q_network(x: Array) -> Array:
        return hk.nets.MLP([256, 2])(x)


    if __name__ == '__main__':
        rl = RLib(
            agent_type=QLearning,
            agent_params={
                'q_network': q_network,
                'optimizer': optax.rmsprop(3e-4, decay=0.95, eps=1e-2),
            },
            ext_type=Gymnasium,
            ext_params={'env_id': 'CartPole-v1'}
        )

        for epoch in range(300):
            env = gym.make('CartPole-v1', render_mode='human')

            _, _ = env.reset()
            action = env.action_space.sample()
            terminal = False

            while not terminal:
                env_state = env.step(action.item())
                action = rl.sample(*env_state)
                terminal = env_state[2] or env_state[3]

After the necessary imports, we create an instance of the ``RLib`` class. We provide the chosen
agent type and the appropriate extension for the problem. This extension will help the agent to infer necessary
information from the environment. Next create a Gymnasium environment and define the training loop. Inside the loop,
we call the ``sample`` method which passes the observations to the agent, updates the agent's internal state
and returns an action proposed by the agent's policy. We apply the returned action in the environment to get its
altered state. We encourage you to see the :ref:`API <api_page>` section for more details about the ``RLib`` class.

Note that in the example above, we use the deep reinforcement learning agent. Our library provides a wide range of
agents, including both deep and classic reinforcement learning agents. To learn more about the available agents,
check out the :ref:`Agents <agents_page>` section. You can also create your own agent. To learn more about creating
custom agents, check out the :ref:`Custom agents <custom_agents>` section.

The extensions are also a crucial part of the Reinforced-lib. You can use the built-in extensions listed in the
:ref:`Extensions <extensions_page>` section, but we highly encourage you to create your own extensions. To learn more
about extensions check out the :ref:`Custom extensions <custom_extensions>` section.

Training and inference modes
----------------------------

Reinforced-lib provides two modes of operation: training and inference. The training mode is the default one. It
enables the agent to learn from the environment. The inference mode is used to evaluate the agent's performance
or to use the agent in the production environment. To use the inference mode, you have to set the ``is_training``
flag to ``False`` in the ``sample`` method:

.. code-block:: python

    action = rl.sample(*env_state, is_training=False)


Logging
-------

The logging module provides a simple yet powerful API for visualizing and analyzing the running algorithm or watching
the training process. You can monitor any observations passed to the agent, the agent's state, and the basic metrics in
real time. If you want to learn more about the logging module, check out the :ref:`Custom loggers <custom_loggers>`
section.

Basic logging
~~~~~~~~~~~~~

Below is the simplest example of using one of the built-in loggers:

.. code-block:: python

    rl = RLib(
        ...
        logger_types=TensorboardLogger,
        logger_sources='cumulative'
    )

In the example above, we use ``TensorboardLogger`` to print the cumulative reward of the agent. The ``logger_sources``
parameter specifies the predefined source of the logger. The source is a name of the observation, the agent's state,
or the metric. `TensorBoard <https://www.tensorflow.org/tensorboard>`_ is a powerful visualization toolkit that
allows you to monitor the training process in real time, create interactive visualizations, and save the logs for later
analysis. You can use the ``TensorboardLogger`` along with other loggers built into Reinforced-lib. To learn more about
available loggers, check out the :ref:`Logging module <logging_page>` section.

Advanced logging
~~~~~~~~~~~~~~~~

You can easily change the logger type, add more sources, and customize the parameters of the logger:

.. code-block:: python

    rl = RLib(
        ...
        logger_types=PlotsLogger,
        logger_sources=['terminal', 'epsilon', ('action', SourceType.METRIC)],
        logger_params={'plots_smoothing': 0.9}
    )

Note that ``terminal`` is the observation name, ``epsilon`` is name of the state of the ``QLearning`` agent,
and ``action`` is the name of the metric. You can mix sources names as long as it does not lead to inconclusiveness.
In the example above, it can be seen that ``action`` is both the name of the observation and the metric. In this case,
you have to write the source name as a tuple containing a name and the type of the source ``(str, SourceType)``
as in the code above.

You can also plug multiple loggers to one source:

.. code-block:: python

    rl = RLib(
        ...
        logger_types=[StdoutLogger, CsvLogger, PlotsLogger],
        logger_sources='cumulative'
    )

Or mix different loggers and sources:

.. code-block:: python

    rl = RLib(
        ...
        logger_types=[StdoutLogger, CsvLogger, PlotsLogger],
        logger_sources=['terminal', 'epsilon', ('action', SourceType.METRIC)],
    )

In this case remember to provide a list of loggers that has the same length as the list of sources, because given
loggers will be used to log values for consecutive sources.

Custom logging
~~~~~~~~~~~~~~

Note that you are not limited to logging only the predefined sources. You can log any value you want. To do this,
you can use the ``log`` method of the ``RLib`` class. Remember to define a logger that will be used to log the value.
You can do this by providing the only logger type (without sources) or by providing the logger type and the source
set to ``None``:

.. code-block:: python

    rl = RLib(
        ...
        logger_types=TensorboardLogger
    )

    rl.log('my_value', 42)

It is possible to mix predefined sources with custom ones:

.. code-block:: python

    rl = RLib(
        ...
        logger_types=[TensorboardLogger, PlotsLogger, StdoutLogger],
        logger_sources=[None, None, ('reward', SourceType.METRIC)]
    )

    rl.log('my_value', 42)

Of course, you can also create your own logger. To learn more about creating custom loggers, check out the
:ref:`Custom loggers <custom_loggers>` section.


Saving experiments
------------------

The ``RLib`` class provides an API for saving your experiment in a compressed ``.lz4`` format.
You can later reconstruct the experiment state and continue from the exact point where you ended or you can
alter some training parameters during the reloading process.


Full reconstruction
~~~~~~~~~~~~~~~~~~~

We can imagine a scenario where we set up the experiment, perform a little training, and then we need to take a break.
Therefore, we save the experiment at the latest state that we would later want to carry on from. When we are ready to continue
with the training, we load the whole experiment to a new RLib instance.

.. code-block:: python

    from reinforced_lib import RLib

    # Setting up the experiment
    rl = RLib(...)

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
Let us recall the above example and modify it a little. We now want to modify on-the-fly the learning rate of the
optimizer:

.. code-block:: python

    from reinforced_lib import RLib
    from reinforced_lib.agents.deep import QLearning
    from reinforced_lib.exts import Gymnasium

    # Setting up the experiment
    rl = RLib(
        agent_type=QLearning,
        agent_params={
            'q_network': q_network,
            'optimizer': optax.adam(1e-3),
        },
        ext_type=Gymnasium,
        ext_params={'env_id': 'CartPole-v1'}
    )

    # Do some training
    # ...

    # Saving experiment state for later
    rl.save("<checkpoint-path>")

    # Load the saved training with altered parameters
    rl = RLib.load(
        "<checkpoint-path>",
        agent_params={
            'q_network': q_network,
            'optimizer': optax.adam(1e-4),
        }
    )

    # Continue the training with new parameters
    # ...

You can change as many parameters as you want. The provided example is constrained only to the agent's
parameter alteration, but you can modify the extension's parameters in the same way. You can even completely
modify the loggers behaviour by providing new ones in ``logger_types``, ``logger_sources`` and ``logger_params``.


Automatic checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~

The ``RLib`` class provides an API for automatic checkpointing. You can specify the frequency of
saving the experiment state and the path to the directory where the checkpoints will be saved. The checkpointing
mechanism is based on the ``save()`` method, so you can use the same API for reloading the experiment.

.. code-block:: python

    rl = RLib(
        ...
        auto_checkpoint=5,
        auto_checkpoint_path="<checkpoint-dir>"
    )

    # Do some training
    # ...

    # Load the saved training
    rl = RLib.load("<checkpoint-path>")

The ``auto_checkpoint`` parameter specifies the frequency of saving the experiment state (in this case every 5th update
of the agent). The ``auto_checkpoint_path`` parameter specifies the path to the directory where the checkpoints will be
saved.


Export to TF Lite
-----------------

Reinforced-lib offers a convenient API to export the agent to the `TensorFlow Lite <https://www.tensorflow.org/lite>`_
format, allowing seamless integration with embedded devices or deployment to production environments.

Exporting the agent
~~~~~~~~~~~~~~~~~~~

To export model you can leverage the  ``to_tflite`` method of the ``RLib`` class:

.. code-block:: python

    rl.to_tflite("<model-path>")

By default, the exported model will include the core functionalities of the agent, namely the ``init``, ``update``,
and ``sample`` methods. It's important to note that the ``init`` method will initialize the basic state of the agent.
For deep learning agents, this means the neural network weights will be randomly initialized, while for multi-armed
bandit agents, the state will be filled with default values.

Exporting with trained state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you wish to export the agent with the state of a specific trained agent, you can achieve this by providing the
``agent_id`` parameter:

.. code-block:: python

    rl.to_tflite("<model-path>", agent_id="<agent-id>")

By specifying the ``agent_id`` parameter, the exported model will be initialized with the state of the corresponding
agent.

Exporting for inference mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, you might only need the agent for inference purposes. To export the agent for inference mode, you need
to set the ``sample_only`` flag to ``True`` and provide the relevant ``agent_id`` parameter:

.. code-block:: python

    rl.to_tflite("<model-path>", agent_id="<agent-id>", sample_only=True)

In this scenario, the exported model will only contain the ``init`` and ``sample`` methods of the agent, and the
``init`` method will return the state of the specified agent.

Requirements
~~~~~~~~~~~~

.. note::

    To export the agent to the TensorFlow Lite format, the ``tensorflow`` package is required. To install the
    package, run the following command:

    .. code-block:: bash

        pip install tensorflow

All built-in agents are adapted to the seamless export to the TensorFlow Lite format. If you want to export a custom
agent, you need to implement the ``update_observation_space`` and ``sample_observation_space`` methods.  Although not
mandatory, we strongly encourage their implementation as they allow easy sampling of the parameters of the agent's
methods. To learn more about the agent's methods, check out the :ref:`Custom agents <custom_agents>` section.


Real-world examples
-------------------

We provide a set of examples that show how to use Reinforced-lib in real-world problems. We highly encourage you to
check them out. You can find them in the `examples directory <https://github.com/m-wojnar/reinforced-lib/tree/main/examples>`_.
