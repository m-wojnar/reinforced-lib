.. _extensions:

Environment extensions
======================

The environment extension is our functionality that allows agent to infer latent observations that are
not originally supported by the environment. You can either choose one of our built-in extensions or
implement your own with the help of this short guide.

.. _key_concepts:

Key concepts
------------

The main axis of this module is the :ref:`abstract class <base_ext>` ``BaseExt``, which provides the core
functionality of extensions. It implements important methods, such as:

* ``get_agent_params`` - this method is responsible for providing default arguments specified by extension
  while creating new agent,
* ``transform`` - composes observations passed by the user with values calculated or defined by the extension
  to create set of parameters required by agent functions. In other words, ``transform`` fills missing
  values and returns full set of required parameters,
* ``setup_transformations`` - based on the update and sample observation spaces of the agent, method creates
  functions that combine observations passed by the user with values calculated by the extension which are
  used by the ``transform`` method.

Above methods are very useful, because they simplify the way we use this library. There are two main benefits
of using extensions:

#. Automatic initialization of agents - en extensions can provide default arguments that can be used to
   initialize an agent. For example, if we would like to create the :ref:```wifi.ParticleFilter`` <particle-filter_agent>`
   agent without using any extension, we would probably do it in the following way:

   .. code-block:: python

       rl = rfl.RLib(
           agent_type=ParticleFilter,
           agent_params={
               'n_mcs': 12,
               'min_snr': 0.0,
               'max_snr': 40.0,
               'initial_power': 16.0206
           }
       )

   On the other hand, if we decide to use the :ref:`IEEE 802.11ax <ieee_802_11_ax>` extension, this parameters can
   be automatically provided by the extension:

   .. code-block:: python

       rl = rfl.RLib(
           agent_type=ParticleFilter,
           ext_type=IEEE_802_11_ax
       )

   We can also overwrite all of only part of default values provided by the extension:

   .. code-block:: python

       rl = rfl.RLib(
           agent_type=ParticleFilter,
           agent_params={
               'initial_power': 21.0
           }
           ext_type=IEEE_802_11_ax,
       )

#. Filling missing parameters - some of the parameters required by the agent can be filled with known values or
   calculated based on a set of basic observations. For example, a ``sample`` method of the
   :ref:```wifi.ParticleFilter`` <particle-filter_agent>` agent requires transmission data rates and minimal SNR
   values required for a successful transmission for each MCS. This values can be found in the IEEE 802.11ax standard
   documentation or precalculated empirically. Below is a sample code that could be used to sample from the agent
   without using any extension:

   .. code-block:: python

       observations = {
           'time': 1.8232,
           'action': 11,
           'n_successful': 10,
           'n_failed': 0,
           'power': 16.0206,
           'cw': 15,
           'rates': jnp.array([7.3, 14.6, 21.9, 29.3, 43.9, 58.5, 65.8, 73.1, 87.8, 97.5, 109.7, 121.9]),
           'min_snrs': jnp.array([0.5, 3.4, 6.5, 9.4, 13.1, 16.9, 18.9, 20.6, 24.1, 25.8, 31.7, 33.7]),
       }
       action = rl.sample(**observations)

   If we use the :ref:`IEEE 802.11ax <ieee_802_11_ax>` extension, part of this parameters can be provided by the
   extension:

   .. code-block:: python

       observations = {
           'time': 1.8232,
           'mcs': 11,
           'n_successful': 10,
           'n_failed': 0,
           'power': 16.0206,
           'cw': 15
       }
       action = rl.sample(**observations)

   We can also overwrite values provided by the extension:

   .. code-block:: python

       observations = {
           'time': 1.8232,
           'mcs': 11,
           'n_successful': 10,
           'n_failed': 0,
           'power': 16.0206,
           'cw': 15,
           'min_snrs': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
       }
       action = rl.sample(**observations)

Default values or functions that calculates missing parameters can be defined using **observation functions**
and **parameter functions**. These functions are decorated with the ``@observation`` and ``@parameter`` decorators
accordingly. More detailed description of this decorator can be found in :ref:`the section <custom_exts>` below.

.. _custom_exts:

Custom extensions
-----------------

To create your own extension, you should inherit from the :ref:`abstract class <base_ext>` ``BaseExt``.
We will present adding custom extension on an example of the :ref:`IEEE 802.11ax <ieee_802_11_ax>` extension.

.. code-block:: python

    class IEEE_802_11_ax(BaseExt)
    
Firstly, we must specify the observation space of the extension. It is a basic set of environment observations
that can be used by the agent and the extension itself to compute missing values. Note that complete set of all
parameters is not necessarily required to use the extension - if agent does not require a given parameter and
it is not used to compute missing values, the extension will ignore it. In the case of the IEEE 802.11ax
environment, the observation space can look like this:

.. code-block:: python

    observation_space = gym.spaces.Dict({
        'time': gym.spaces.Box(0.0, np.inf, (1,)),
        'n_successful': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'n_failed': gym.spaces.Box(0, np.inf, (1,), np.int32),
        'n_wifi': gym.spaces.Box(1, np.inf, (1,), np.int32),
        'power': gym.spaces.Box(-np.inf, np.inf, (1,)),
        'cw': gym.spaces.Discrete(32767),
        'mcs': gym.spaces.Discrete(12)
    })

Next, we define the **parameter function** that will provide the initial power value for agents that require
this parameter as a constructor argument. We can do this by creating an appropriate method and decorating it with
the ``@parameter`` decorator. Parameter function are methods of the extension class and cannot take any additional
arguments:

.. code-block:: python

    @parameter
    def initial_power(self):
        return 16.0206

We can also specify type of the returned value in the OpenAI Gym format. It will help the library to check if
a given value type is compatible with the argument type required by the agent:

.. code-block:: python

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def initial_power(self) -> float:
        return 16.0206

Note that name of the function must match name of the argument required by the agent. If there already exists
a function with that name, we can name the function differently and explicitly define the argument name in
the decorator:

.. code-block:: python

    @parameter(parameter_name='initial_power', parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def default_pow(self) -> float:
        return 16.0206

We define the **observation functions** analogous to parameter functions. The only differences are that we use
the ``@observation`` decorator and that the implemented method takes additional parameters. Below is an
example observation function that provides approximated collision probability in dense IEEE 802.11ax networks:

.. code-block:: python

    @observation
    def pc(self, n_wifi, *args, **kwargs):
        return 0.154887 * np.log(1.03102 * n_wifi)

Note that the observation function can take parameters that are specified in the observation space.
:ref:`BaseExt <base_ext>` methods will automatically pass the given observation to the function to allow
dynamic computation of the returned value. What is important, observation methods take ``*args`` and ``**kwargs``
as the last parameters (this is required by the internal behaviour of the ``setup_transformations`` function).
As previously, name of the function should match name of the filled parameter, but we can specify parameter name
and returned type in the decorator:

.. code-block:: python

    @observation(observation_name='pc', observation_type=gym.spaces.Box(0.0, 1.0, (1,)))
    def collision_probability(self, n_wifi: int, *args, **kwargs) -> float:
        return 0.154887 * np.log(1.03102 * n_wifi)

Full source code of the IEEE 802.11ax extension can be found :ref:`here <https://github.com/m-wojnar/reinforced-lib/blob/main/reinforced_lib/exts/ieee_802_11_ax.py>`.

Rules and limitations
---------------------

Extensions are very powerful mechanism that makes the Reinforced-lib universal and easy to use. The ``BaseExt``
methods can handle complex and nested observation spaces, such as the
:ref:`example ones <https://github.com/m-wojnar/reinforced-lib/blob/main/tests/extqs/base_ext_test.py>`.
However, there are some rules and limitations that programmers and users must take into consideration:

* arguments and parameters provided by the user have higher priority than default or calculated values provided
  by the extension,
* parameter functions cannot take any arguments (except ``self``),
* you cannot use extension with a given agent if the agent requires a parameter that is not listed in the
  extensions observation space or cannot be provided by an observation function - you have to add an observation
  to the observation space, implement appropriate observation function of use the agent without any extension,
* missing parameters filling is supported only if the type of the extension observation space and the type of agent
  spaces can be matched - that means they both must be:
  * a dict type - ``gym.spaces.Dict``,
  * a "simple" type - ``gym.spaces.Box``, ``gym.spaces.Discrete``, ``gym.spaces.MultiBinary``, ``gym.spaces.MultiDiscrete``, ``gym.spaces.Space``,
* missing parameters filling is not supported if spaces inherit from ``gym.spaces.Tuple`` - values would have
  to be matched based on the type and this can lead to ambiguities if there are multiple parameters with the same type,
* if spaces do not inherit from ``gym.spaces.Dict``, missing values are matched based on the type of the value,
  not the name - first function that type matches the agent space is choosen,
* if an observation function requires some parameter and it is not provided by a named argument, ``BaseExt`` will
  select the first (possibly nested) positional argument and pass it to the function, but if there will be no
  positional arguments, library will raise an exception.

.. _base_ext:

BaseExt
-------

.. currentmodule:: reinforced_lib.exts.base_ext

.. autoclass:: BaseExt
    :members:

List of extensions
------------------

.. _ieee_802_11_ax:

IEEE 802.11ax
~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.exts.ieee_802_11_ax

.. autoclass:: IEEE_802_11_ax
    :show-inheritance:
    :members:
