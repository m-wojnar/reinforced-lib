.. _custom_extensions:

Custom extensions
=================

The extensions is a unique functionality that allows a library to infer missing observations that are
not originally supported by the environment. You can either choose one of our built-in extensions or
implement your own with the help of this short guide.


Key concepts of extensions
--------------------------

There are three main benefits of using extensions:

#. Automatic initialization of agents - an extension can provide default arguments that can be used to
   initialize an agent. For example, if we would like to train the :ref:`deep Double Q-learning agent
   <Deep Double Q-Learning (DQN)>` on a `cart-pole` environment without using any extension, we would
   probably do it in the following way:

   .. code-block:: python

       rl = RLib(
           agent_type=QLearning,
           agent_params={
               'q_network': q_network,
               'obs_space_shape': (4,),
               'act_space_size': 2
           },
           no_ext_mode=True
       )

   On the other hand, if we decide to use the :ref:`Gymnasium extension <Gymnasium>`,
   some of the parameters can be automatically provided by the extension:

   .. code-block:: python

       rl = RLib(
           agent_type=QLearning,
           agent_params={'q_network': q_network}
           ext_type=Gymnasium,
           ext_params={'env_id': 'CartPole-v1'},
       )

   We can also overwrite all or only part of the default values provided by the extension:

   .. code-block:: python

       rl = RLib(
           agent_type=QLearning,
           agent_params={
               'q_network': q_network,
               'act_space_size': 3
           },
           ext_type=Gymnasium,
           ext_params={'env_id': 'CartPole-v1'},
       )

#. Simplification of parameter passing - extensions allow automatic matching observations returned by the environment
   to the appropriate methods of the agent. The code snippet below shows the agent and environment interaction loop
   without using any extension:

   .. code-block:: python

    while not terminal:
        env_state, reward, terminal, truncated, info = env.step(action.item())

        action = rl.sample(
            update_observations={
                'env_state': env_state,
                'action': action,
                'reward': reward,
                'terminal': terminal
            },
            sample_observations={
                'env_state': env_state
            }
        )

   The following code is equivalent to the above but makes use of the properly defined
   :ref:`Gymnasium extension <Gymnasium>`:

   .. code-block:: python

    while not env_state[2]:
        env_state = env.step(action.item())
        action = rl.sample(*env_state)

#. Filling missing parameters - some parameters required by the agent can be filled with known values or
   calculated based on a set of basic observations. For example, a ``sample`` method of the :ref:`Thompson
   sampling <Thompson sampling>` agent requires a context vector. As it is a domain specific knowledge,
   these values can be found in the appropriate extension. Below is a sample code that could be used to sample
   the next action in the IEEE 802.11ax rate adaptation problem without using any extension:

   .. code-block:: python

        rl = RLib(
            agent_type=ThompsonSampling,
            agent_params={'n_arms': 12},
            no_ext_mode=True
        )

       observations = {
           'delta_time': 0.18232,
           'n_successful': 10,
           'n_failed': 0,
           'context': np.array(
               [7.3, 14.6, 21.9, 29.3, 43.9, 58.5,
               65.8, 73.1, 87.8, 97.5, 109.7, 121.9]
           )
       }
       action = rl.sample(**observations)

   If we use the `IEEE 802.11ax RA extension <https://github.com/m-wojnar/reinforced-lib/blob/main/examples/ns-3-ra/ext.py>`_,
   part of these parameters can be provided by the extension:

   .. code-block:: python

        rl = RLib(
            agent_type=ThompsonSampling,
            ext_type=IEEE_802_11_ax_RA
        )

       observations = {
           'delta_time': 0.18232,
           'n_successful': 10,
           'n_failed': 0
       }
       action = rl.sample(**observations)

   We can also overwrite the values provided by the extension:

   .. code-block:: python

       observations = {
           'delta_time': 0.18232,
           'n_successful': 10,
           'n_failed': 0,
           'context': jnp.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
       }
       action = rl.sample(**observations)

You can define default values as initialization arguments for agents through parameter functions. Additionally,
default values or functions to calculate missing observations can be defined using observation functions. To designate
these functions correctly, they are decorated with the ``@observation`` and ``@parameter`` decorators respectively.
A more detailed description of this decorator can be found in :ref:`the section below <Implementing new extensions>`.


Implementing new extensions
---------------------------

To create your own extension, you should inherit from the :ref:`abstract class <BaseExt>` ``BaseExt``. We
present adding a custom extension using an example of the extension used in the
`IEEE 802.11ax rate adaptation <https://github.com/m-wojnar/reinforced-lib/blob/main/examples/ns-3-ra/ext.py>`_, problem.

.. code-block:: python

    class IEEE_802_11_ax_RA(BaseExt)
    
First, we must specify the observation space of the extension. It is a basic set of environment observations
that can be used by the extension to compute missing values. Note that a complete set of all parameters is not
necessarily required to use the extension - if an agent does not require a given parameter and it is not used to
compute missing values, the extension will ignore it. In the case of the IEEE 802.11ax environment, the observation
space can look like this:

.. code-block:: python

    observation_space = gym.spaces.Dict({
        'time': gym.spaces.Box(0.0, np.inf, (1,)),
        'n_successful': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'n_failed': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'n_wifi': gym.spaces.Box(1, np.inf, (1,), np.int32),
        'power': gym.spaces.Box(-np.inf, np.inf, (1,)),
        'cw': gym.spaces.Discrete(32767)
    })

Next, we define the *parameter function* that will provide the default power value for agents that require
this parameter as a constructor argument. We can do this by creating an appropriate method and decorating it with
the ``@parameter`` decorator. The *parameter functions* are methods of the extension and cannot take any additional
arguments:

.. code-block:: python

    @parameter()
    def default_power(self):
        return 16.0206

We can also specify the type of the returned value in `Gymnasium <https://gymnasium.farama.org/>`_ (former OpenAI Gym)
format. It will help the library to check if a given value type is compatible with the argument required by the agent:

.. code-block:: python

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def default_power(self) -> float:
        return 16.0206

Note that the name of the function must match the name of the argument required by the agent. If there already exists
a function with that name, we can name the function differently and explicitly define the argument name in
the decorator:

.. code-block:: python

    @parameter(parameter_name='default_power', parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def default_pow(self) -> float:
        return 16.0206

We define the *observation functions* by analogy to *parameter functions*. The differences are that we use
the ``@observation`` decorator and that the implemented methods can take additional parameters. Below is an
example *observation function* that provides a reward calculated as an approximated throughput in the IEEE 802.11ax
environment:

.. code-block:: python

    @observation()
    def reward(self, mcs, n_successful, n_failed, *args, **kwargs):
        if n_successful + n_failed > 0:
            return self._wifi_modes_rates[mcs] * n_successful / (n_successful + n_failed)
        else:
            return 0.0

Note that the *observation function* can take parameters that are specified in the observation space.
``BaseExt`` will automatically pass the given observation to the function to allow dynamic computation of the
returned value. What is important, observation methods must take ``*args`` and ``**kwargs`` as the last parameters
(this is required by the internal behavior of the ``setup_transformations`` function). As previously, the name of
the function should match the name of the filled parameter, but we can specify the parameter name and returned
type in the decorator:

.. code-block:: python

    @observation(observation_name='reward', observation_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def custom_reward(self, mcs: int, n_successful: int, n_failed: int, *args, **kwargs) -> float:
        if n_successful + n_failed > 0:
            return self._wifi_modes_rates[mcs] * n_successful / (n_successful + n_failed)
        else:
            return 0.0


Template extension
------------------

To simplify the process of creating new extensions, we provide an example extension that can be used as a
starting point for creating your own extensions. The IEEE 802.11ax rate adaptation extension can be found `here <https://github.com/m-wojnar/reinforced-lib/blob/main/examples/ns-3-ra/ext.py>`_:

.. code-block:: python

    import gymnasium as gym
    import numpy as np

    from reinforced_lib.exts import BaseExt, observation, parameter


    class IEEE_802_11_ax_RA(BaseExt):
        def __init__(self) -> None:
            super().__init__()
            self.last_time = 0.0

        observation_space = gym.spaces.Dict({
            'time': gym.spaces.Box(0.0, np.inf, (1,)),
            'n_successful': gym.spaces.Box(0, np.inf, (1,), np.int32),
            'n_failed': gym.spaces.Box(0, np.inf, (1,), np.int32),
            'n_wifi': gym.spaces.Box(1, np.inf, (1,), np.int32),
            'power': gym.spaces.Box(-np.inf, np.inf, (1,)),
            'cw': gym.spaces.Discrete(32767)
        })

        _wifi_modes_rates = np.array([
            7.3, 14.6, 21.9, 29.3, 43.9, 58.5,
            65.8, 73.1, 87.8, 97.5, 109.7, 121.9
        ], dtype=np.float32)

        @observation(observation_type=gym.spaces.Box(0.0, np.inf, (len(_wifi_modes_rates),)))
        def rates(self, *args, **kwargs) -> np.ndarray:
            return self._wifi_modes_rates

        @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (len(_wifi_modes_rates),)))
        def context(self, *args, **kwargs) -> np.ndarray:
            return self.rates()

        @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
        def reward(self, action: int, n_successful: int, n_failed: int, *args, **kwargs) -> float:
            if n_successful + n_failed > 0:
                return self._wifi_modes_rates[action] * n_successful / (n_successful + n_failed)
            else:
                return 0.0

        @observation(observation_type=gym.spaces.Box(0.0, np.inf, (1,)))
        def delta_time(self, time: float, *args, **kwargs) -> float:
            delta_time = time - self.last_time
            self.last_time = time
            return delta_time

        @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (6,)))
        def env_state(
                self, time: float, n_successful: int, n_failed: int,
                n_wifi: int, power: float, cw: int, *args, **kwargs
        ) -> np.ndarray:
            return np.array([self.delta_time(time), n_successful, n_failed, n_wifi, power, cw], dtype=np.float32)

        @observation(observation_type=gym.spaces.MultiBinary(1))
        def terminal(self, *args, **kwargs) -> bool:
            return False

        @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
        def n_mcs(self) -> int:
            return len(self._wifi_modes_rates)

        @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
        def n_arms(self) -> int:
            return self.n_mcs()

        @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
        def default_power(self) -> float:
            return 16.0206

        @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
        def min_reward(self) -> float:
            return 0

        @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
        def max_reward(self) -> int:
            return self._wifi_modes_rates.max()

        @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, np.inf, (1,), np.int32)))
        def obs_space_shape(self) -> tuple:
            return tuple((6,))

        @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, np.inf, (1,), np.int32)))
        def act_space_shape(self) -> tuple:
            return tuple((1,))

        @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
        def act_space_size(self) -> int:
            return 12

        @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(-np.inf, np.inf)))
        def min_action(self) -> tuple:
            return 0

        @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(-np.inf, np.inf)))
        def max_action(self) -> tuple:
            return 11


Rules and limitations
---------------------

Extensions are powerful mechanisms that make Reinforced-lib easy to use. The ``BaseExt`` methods can handle
complex and nested observation spaces, such as these
`example ones <https://github.com/m-wojnar/reinforced-lib/blob/main/test/exts/test_base_ext.py>`_.
However, there are some rules and limitations that programmers and users must consider:

* arguments and parameters provided by the user have higher priority than the default or calculated by the extension,
* *parameter functions* cannot take any arguments (except ``self``),
* you cannot use an extension with a given agent if the agent requires a parameter that is not listed in the
  extensions observation space or cannot be provided by an *observation function* - you have to add an observation
  to the observation space, implement the appropriate *observation function* or use the agent without any extension,
* missing parameter filling is supported only if the type of the extension observation space and the type of agent
  space can be matched - that means they both must be:

  * a dict type - ``gym.spaces.Dict``,
  * or a "simple" type - ``gym.spaces.Box``, ``gym.spaces.Discrete``, ``gym.spaces.MultiBinary``, ``gym.spaces.MultiDiscrete``, ``gym.spaces.Space``,

* missing parameter filling is not supported if spaces inherit from ``gym.spaces.Tuple`` - values would have
  to be matched based on the type and this can lead to ambiguities if there are multiple parameters with the same type,
* if spaces do not inherit from ``gym.spaces.Dict``, missing values are matched based on the type of the value,
  not the name, so the first function that type matches the agent space is chosen,
* if an *observation function* requires some parameter and it is not provided by a named argument, ``BaseExt`` will
  select the first (possibly nested) positional argument and pass it to the function, but if there are no
  positional arguments, the library will raise an exception.


How do extensions work?
-----------------------

The main axis of this module is the :ref:`abstract class <BaseExt>` ``BaseExt``, which provides the core
functionality of extensions. It implements important methods, such as ``get_agent_params``, ``transform``,
and ``setup_transformations``. The class internally makes use of these methods to provide a simple
and powerful API of Reinforced-lib. You can read more about the ``BaseExt`` class :ref:`here <BaseExt>`
or check out `the source code <https://github.com/m-wojnar/reinforced-lib/blob/main/reinforced_lib/exts/base_ext.py>`_.

