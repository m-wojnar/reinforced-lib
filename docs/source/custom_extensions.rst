.. _custom_extensions:

Custom extensions
=================

The environment extension is our functionality that allows an agent to infer latent observations that are
not originally supported by the environment. You can either choose one of our built-in extensions or
implement your own with the help of this short guide.


Key concepts
------------

There are three main benefits of using extensions:

#. Automatic initialization of agents - an extension can provide default arguments that can be used to
   initialize an agent. For example, if we would like to create the ``wifi.ParticleFilter``
   :ref:`agent <Particle Filter (Wi-Fi)>` without using any extension, we would probably do it in the
   following way:

   .. code-block:: python

       rl = rfl.RLib(
           agent_type=ParticleFilter,
           agent_params={
               'default_power': 16.0206
           }
       )

   On the other hand, if we decide to use the ``IEEE_802_11_ax`` :ref:`extension <IEEE 802.11ax>` extension,
   this parameters can be automatically provided by the extension:

   .. code-block:: python

       rl = rfl.RLib(
           agent_type=ParticleFilter,
           ext_type=IEEE_802_11_ax_RA
       )

   We can also overwrite all or only part of the default values provided by the extension:

   .. code-block:: python

       rl = rfl.RLib(
           agent_type=ParticleFilter,
           agent_params={
               'default_power': 16.0206
           }
           ext_type=IEEE_802_11_ax_RA,
       )

#. Simplification of parameter passing - extensions allow automatic matching observations returned by the environment
   to the appropriate methods of the agent. The code snippet below shows the use of the library to select the next
   action without using any extension:

   .. code-block:: python

       action = rl.sample(
           update_observations={
               'action': action,
               'n_successful': ns,
               'n_failed': nf,
               'time': t,
               'power': p,
               'cw': cw
           },
           sample_observations={
               'time': t,
               'power': p,
               'rates': rates
           }
       )

   The following code is equivalent to the above but makes use of the properly defined
   ``IEEE_802_11_ax`` :ref:`extension <IEEE 802.11ax>` extension:

   .. code-block:: python

       action = rl.sample(**observations)

#. Filling missing parameters - some parameters required by the agent can be filled with known values or
   calculated based on a set of basic observations. For example, a ``sample`` method of the ``wifi.ParticleFilter``
   :ref:`agent <Particle Filter (Wi-Fi)>` requires transmission data rates for each MCS. These values can be found in
   the IEEE 802.11ax standard documentation. Below is a sample code that could be used to sample the next action from
   the agent without using any extension:

   .. code-block:: python

       observations = {
           'time': 1.8232,
           'action': 11,
           'n_successful': 10,
           'n_failed': 0,
           'power': 16.0206,
           'cw': 15,
           'rates': jnp.array([7.3, 14.6, 21.9, 29.3, 43.9, 58.5, 65.8, 73.1, 87.8, 97.5, 109.7, 121.9])
       }
       action = rl.sample(**observations)

   If we use the ``IEEE_802_11_ax`` :ref:`extension <IEEE 802.11ax>` extension, part of these parameters can be
   provided by the extension:

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

   We can also overwrite the values provided by the extension:

   .. code-block:: python

       observations = {
           'time': 1.8232,
           'mcs': 11,
           'n_successful': 10,
           'n_failed': 0,
           'power': 16.0206,
           'cw': 15,
           'rates': jnp.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
       }
       action = rl.sample(**observations)

Default values or functions that calculate missing parameters can be defined using *observation functions*
and *parameter functions*. These functions are decorated with the ``@observation`` and ``@parameter`` decorators
accordingly. A more detailed description of this decorator can be found in :ref:`the section below <Customizing extensions>`.


Customizing extensions
----------------------

To create your own extension, you should inherit from the :ref:`abstract class <BaseExt>` ``BaseExt``. We
present adding a custom extension using an example of the ``IEEE_802_11_ax`` :ref:`extension <IEEE 802.11ax>` extension.

.. code-block:: python

    class IEEE_802_11_ax(BaseExt)
    
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
        'cw': gym.spaces.Discrete(32767),
        'mcs': gym.spaces.Discrete(12)
    })

Next, we define the *parameter function* that will provide the default power value for agents that require
this parameter as a constructor argument. We can do this by creating an appropriate method and decorating it with
the ``@parameter`` decorator. The *parameter functions* are methods of the extension and cannot take any additional
arguments:

.. code-block:: python

    @parameter()
    def default_power(self):
        return 16.0206

We can also specify the type of the returned value in the OpenAI Gym format. It will help the library to check if
a given value type is compatible with the argument required by the agent:

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

The full source code of the IEEE 802.11ax extension can be found `here <https://github.com/m-wojnar/reinforced-lib/blob/main/reinforced_lib/exts/ieee_802_11_ax.py>`_.


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

