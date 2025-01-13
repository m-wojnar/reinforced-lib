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

    Since we tested the Reinforced-lib on Python 3.12, we recommend using Python 3.12+.


Basic usage
-----------

The vital benefit of using Reinforced-lib is a simplification of the agent-environment interaction loop. Thanks to the
``Rlib`` :ref:`class <RLib Class>`, the initialization of the agent and passing the environment state to the agent are
significantly simplified. Below, we present the basic training loop with the simplifications provided by Reinforced-lib.

.. code-block:: python

    import gymnasium as gym
    import optax
    from chex import Array
    from flax import linen as nn

    from reinforced_lib import RLib
    from reinforced_lib.agents.deep import DQN
    from reinforced_lib.exts import Gymnasium


    class QNetwork(nn.Module):
        @nn.compact
        def __call__(self, x: Array) -> Array:
            x = nn.Dense(256)(x)
            x = nn.relu(x)
            return nn.Dense(2)(x)


    if __name__ == '__main__':
        rl = RLib(
            agent_type=DQN,
            agent_params={
                'q_network': QNetwork(),
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
information from the environment. Next create a gymnasium environment and define the training loop. Inside the loop,
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

Interaction with multiple agents
--------------------------------

Reinforced-lib allows you to use multiple agent instances in the same environment. This feature is useful when you want
to train multiple agents in parallel or use multiple agents to solve the same problem. To achieve this, you need to
initialize the instances of the agents by calling the ``init`` method of the ``RLib`` class a certain number of times:

.. code-block:: python

    rl = RLib(..)

    for _ in range(n_agents):
        rl.init()

Reproducibility
~~~~~~~~~~~~~~~

JAX is focused on reproducibility, and it provides a robust pseudo-random number generator (PRNG) that allows you to
control the randomness of the computations. PRNG requires setting the random seed to ensure that the results of the
computation are reproducible. Reinforced-lib provides an API for setting the random seed for the JAX library.
You can set the seed by providing the ``seed`` parameter when creating the instance of the agent:

.. code-block:: python

    rl = RLib(...)
    rl.init(seed=123)

The seed is initially configured as 42 and the ``init`` method is triggered automatically after the first sampling call.
It eliminates the need to manually call the ``init`` method unless you want to provide custom seed, thus ensuring
reproducibility.

.. note::

    Remember that the reproducibility of the computations is guaranteed only for the agents from Reinforced-lib.
    You have to ensure that the environment you use is reproducible as well.

Loggers
-------

The loggers module provides a simple yet powerful API for visualizing and analyzing the running algorithm or watching
the training process. You can monitor any observations passed to the agent, the agent's state, and the basic metrics in
real time. If you want to learn more about the loggers module, check out the :ref:`Custom loggers <custom_loggers>`
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
available loggers, check out the :ref:`Loggers module <loggers_page>` section.

.. warning::

    Some loggers perform actions upon completion of the training, such as saving the logs, closing the file, or
    uploading the logs to the cloud. Therefore, it is important to gracefully close the Reinforced-lib instance
    to ensure that the logs are saved properly. If you create a Reinforced-lib instance in a function, the destructor
    will be called automatically when the function ends and you do not have to worry about anything. However, if
    you create an instance in the main script, you have to close it manually by calling the ``finish`` method:

    .. code-block:: python

        rl = RLib(...)
        # ...
        rl.finish()

    or by using the ``del`` statement:

    .. code-block:: python

        rl = RLib(...)
        # ...
        del rl

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

Note that ``terminal`` is the observation name, ``epsilon`` is name of the state of the ``DQN`` agent,
and ``action`` is the name of the metric. You can mix sources names as long as it does not lead to inconclusiveness.
In the example above, it can be seen that ``action`` is both the name of the observation and the metric. In this case,
you have to write the source name as a tuple containing a name and the type of the source ``(str, SourceType)``
as in the code above.

You can also plug multiple loggers to output the logs to different destinations simultaneously:

.. code-block:: python

    rl = RLib(
        ...
        logger_types=[StdoutLogger, CsvLogger, PlotsLogger],
        logger_sources=['terminal', 'epsilon', ('action', SourceType.METRIC)],
    )


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
        logger_sources=('reward', SourceType.METRIC)
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

Reinforced-lib can even save the architecture of your agent's neural network. It is possible thanks to the
`cloudpickle <https://github.com/cloudpipe/cloudpickle>`_ library allowing to serialize the flax modules.
However, if you use your own implementation of agents or extensions, you have to ensure that they are available
when you restore the experiment as Reinforced-lib does not save the source code of the custom classes.

.. note::

    Remember that the ``RLib`` class will not save the state of the environment. You have to save the environment
    state separately if you want to continue the training from the exact point where you ended.

.. warning::

    As of today (2024-02-08), cloudpickle does not support the serialization of the custom modules defined outside of
    the main definition. It means that if you implement part of your model in a separate class, you will not be able
    to restore the experiment. We are working on a solution to this problem.

    The temporary solution is to define the whole model in one class as follows:

    .. code-block:: python

        class QNetwork(nn.Module):
            @nn.compact
            def __call__(self, x):
                class MyModule(nn.Module):
                    @nn.compact
                    def __call__(self, x):
                        ...
                        return x

                x = MyModule()(x)
                ...
                return x

    To improve code readability, you can also define modules in external functions and then call them to include
    custom module definitions in the main class. For example:

    .. code-block:: python

        def my_module_fn():
            class MyModule(nn.Module):
                @nn.compact
                def __call__(self, x):
                    ...
                    return x

            return MyModule

        class QNetwork(nn.Module):
            @nn.compact
            def __call__(self, x):
                MyModule = my_module_fn(x)

                x = MyModule()(x)
                ...
                return x


Dynamic parameter change
~~~~~~~~~~~~~~~~~~~~~~~~~

Another feature of the saving mechanism is that it allows us to dynamically change training parameters.
Let us recall the above example and modify it a little. We now want to modify on-the-fly the learning rate of the
optimizer:

.. code-block:: python

    from reinforced_lib import RLib
    from reinforced_lib.agents.deep import DQN
    from reinforced_lib.exts import Gymnasium

    # Setting up the experiment
    rl = RLib(
        agent_type=DQN,
        agent_params={
            'q_network': QNetwork(),
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
            'q_network': QNetwork(),
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


64-bit floating-point precision
-------------------------------

By default, JAX uses 32-bit floating-point precision. However, in some cases, you might want to use 64-bit
floating-point precision. The easiest way to achieve this is to set the ``JAX_ENABLE_X64`` environment variable to
``True``:

.. code-block:: bash

    export JAX_ENABLE_X64=True

Alternatively, you can set the environment variable in your Python script:

.. code-block:: python

    import os
    os.environ['JAX_ENABLE_X64'] = 'True'


Real-world examples
-------------------

To help you get started and learn how to utilize Reinforced-lib in real-world scenarios, we have prepared a
comprehensive set of examples. We strongly encourage you to explore them in the dedicated
:ref:`Examples <examples_page>` section.

To access the source code of these examples, simply navigate to the `examples directory
<https://github.com/m-wojnar/reinforced-lib/tree/main/examples>`_ on our GitHub repository.
