.. _custom_agents:

Custom agents
=============

Although our library provides a palette of already implemented :ref:`agents <Agents>`, you might want to
add a personalised one to the collection. This guide is to help you with this task.


Implementing new agents
-----------------------

To fully benefit from Reinforced-lib features, including JAX jit optimization, your agent
should inherit from the :ref:`abstract class <BaseAgent>` ``BaseAgent``. We present adding a
custom agent on an example of a simple epsilon-greedy agent:

.. code-block:: python

    class EGreedy(BaseAgent)

Firstly, we need to define the state of our agent, which in our case will hold

    * quality values of each arm (`Q`),
    * number of each arms' tries (`N`),,

and will inherit from ``AgentState``:

.. code-block:: python
    
    @dataclass
    class EGreedyState(AgentState):

        Q: Array
        N: Array

The ``BaseAgent`` interface breaks the agent's behaviour into three methods:

    * `init(PRNGKey, ...) -> AgentState` - initializes the agent's state,
    * `update(AgentState, PRNGKey, ...) -> AgentState` - updates the agent's state after performing some action and receiving a reward,
    * `sample(AgentState, PRNGKey, ...) -> Action` - samples new action according to the agent's and environment's state.

We define the Epsilon-greedy agent, which will have 3 static methods:

.. code-block:: python
    
    # This method initializes the agent with 'n_arms' arms 
    @staticmethod
    def init(
        key: PRNGKey,
        n_arms: jnp.int32
    ) -> EGreedyState:

        return EGreedyState(

            # The initial Q values are set as zeros, due to the lack of prior knowledge
            Q=jnp.zeros(n_arms),

            # The numbers of tries are set as ones, to avoid null division in Q value update
            N=jnp.ones(n_arms, dtype=jnp.int32)
        )
    
    # This method updates the agents state
    @staticmethod
    def update(
        state: EGreedyState,
        key: PRNGKey,
        action: jnp.int32,
        reward: Scalar,
    ) -> EGreedyState:

        return EGreedyState(

            # Q value update
            Q=state.Q.at[action].add((reward - state.Q[action]) / state.N[action]),

            # Incrementing the number of tries on appropriate arm
            N=state.N.at[action].add(1)
        )
    
    # This method samples new action according to the agents state (experience)
    @staticmethod
    def sample(
        state: EGreedyState,
        key: PRNGKey,
        e: Scalar
    ) -> jnp.int32:

        # Split PRNGkey to use it twice
        epsilon_key, choice_key = jax.random.split(key)

        # We further want to jax.jit this function, so basic 'if' is not allowed here
        return jax.lax.cond(

            # The agent experiments with probability e
            jax.random.uniform(epsilon_key) < e,

            # On exploration, agent chooses a random arm
            lambda: jax.random.choice(choice_key, state.Q.size),

            # On exploitation, agent chooses the best known arm
            lambda: jnp.argmax(state.Q)
        )

Having defined these static methods, we can implement the class constructor:

.. code-block:: python
    
    def __init__(
        self, 
        n_arms: jnp.int32, 
        e: Scalar
    ) -> None:

        # Make sure that epsilon has correct value
        assert 0 <= e <= 1

        # We specify the features of our agent
        self.n_arms = n_arms

        # Here we can use the jax.jit() functionality with the previously
        # defined behaviour functions, to make the agent super fast.
        # Note that we use partial() to specify the parameters that are
        # constant during the agent's lifetime to avoid passing them
        # every time the function is called.
        self.init = jax.jit(partial(self.init, n_arms=self.n_arms))
        self.update = jax.jit(self.update)
        self.sample = jax.jit(partial(self.sample, e=e))

Now we specify the initialization arguments of our agent (i.e., the parameters that are required by the
agent's constructor). This is done by implementing the static method ``parameter_space()`` which returns
a dictionary in the format of a `Gymnasium <https://gymnasium.farama.org/>`_ space. It is not required
to implement this method, but it is a good practice to do so. This enables the library to automatically
provide initialization arguments specified by :ref:`extensions <Environment extensions>`.

.. code-block:: python

    # Parameters required by the agent constructor in Gymnasium format.
    # Type of returned value is required to be gym.spaces.Dict.
    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'e': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32)
        })

Specifying the action space of the agent is accomplished by implementing the ``action_space`` property.
While not mandatory, adhering to this practice is recommended as it allows users to conveniently inspect
the agent's action space through the ``action_space`` method of the ``RLib`` class.

.. code-block:: python

    # Action returned by the agent in Gymnasium format.
    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(self.n_arms)

Finally, we define the observation spaces for our agent by implementing the properties called
``update_observation_space`` and ``sample_observation_space``. Although not mandatory, we strongly
encourage their implementation as it allows the library to deduce absent values from raw observations
and functions defined in the :ref:`extensions <Environment extensions>`. Moreover, having these properties
implemented facilitates a seamless export of the agent to the TensorFlow Lite format, where
the library can automatically generate an example set of parameters during the export procedure.

.. code-block:: python
    
    # Parameters required by the 'update' method in Gymnasium format.
    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_arms),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), jnp.float32)
        })
    
    # Parameters required by the 'sample' method in Gymnasium format.
    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

Now we have a ready to operate epsilon-greedy agent!


Template agent
--------------

Here is all of the above code in one piece. You can copy-paste it and use as an inspiration
to create your own agent.

.. code-block:: python

    from functools import partial

    import gymnasium as gym
    import jax
    import jax.numpy as jnp
    from chex import dataclass, Array, Scalar, PRNGKey

    from reinforced_lib.agents import BaseAgent, AgentState


    @dataclass
    class EGreedyState(AgentState):
        Q: Array
        N: Array


    class EGreedy(BaseAgent):

        def __init__(
                self,
                n_arms: jnp.int32,
                e: Scalar
        ) -> None:
            assert 0 <= e <= 1

            self.n_arms = n_arms

            self.init = jax.jit(partial(self.init, n_arms=n_arms))
            self.update = jax.jit(self.update)
            self.sample = jax.jit(partial(self.sample, e=e))

        @staticmethod
        def parameter_space() -> gym.spaces.Dict:
            return gym.spaces.Dict({
                'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
                'e': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32)
            })

        @property
        def update_observation_space(self) -> gym.spaces.Dict:
            return gym.spaces.Dict({
                'action': gym.spaces.Discrete(self.n_arms),
                'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), jnp.float32)
            })

        @property
        def sample_observation_space(self) -> gym.spaces.Dict:
            return gym.spaces.Dict({})

        @property
        def action_space(self) -> gym.spaces.Space:
            return gym.spaces.Discrete(self.n_arms)

        @staticmethod
        def init(
                key: PRNGKey,
                n_arms: jnp.int32
        ) -> EGreedyState:

            return EGreedyState(
                Q=jnp.zeros(n_arms),
                N=jnp.ones(n_arms, dtype=jnp.int32)
            )

        @staticmethod
        def update(
            state: EGreedyState,
            key: PRNGKey,
            action: jnp.int32,
            reward: Scalar
        ) -> EGreedyState:

            return EGreedyState(
                Q=state.Q.at[action].add((reward - state.Q[action]) / state.N[action]),
                N=state.N.at[action].add(1)
            )

        @staticmethod
        def sample(
            state: EGreedyState,
            key: PRNGKey,
            e: Scalar
        ) -> jnp.int32:

            epsilon_key, choice_key = jax.random.split(key)

            return jax.lax.cond(
                jax.random.uniform(epsilon_key) < e,
                lambda: jax.random.choice(choice_key, state.Q.size),
                lambda: jnp.argmax(state.Q)
            )


Deep reinforcement learning agents
----------------------------------

Although the above example is a simple one, it is not hard to extend it to deep reinforcement learning (DRL) agents.
This can be achieved by leveraging the JAX ecosystem, along with the `haiku <https://dm-haiku.readthedocs.io/>`_
library, which provides a convenient way to define neural networks, and `optax <https://optax.readthedocs.io/>`_,
which provides a set of optimizers. Below, we provide excerpts of the code for the :ref:`deep Q-learning agent
<Deep Q-Learning>`.

The state of the DRL agent often contains parameters and state of the neural network as well as an experience
replay buffer:

.. code-block:: python

    @dataclass
    class QLearningState(AgentState):
        params: hk.Params
        state: hk.State
        opt_state: optax.OptState

        replay_buffer: ReplayBuffer
        prev_env_state: Array
        epsilon: Scalar

The agent's constructor allows you to specify parameters for the neural network architecture and optimizer, enabling
users to have full control over their choice and enhancing the agent's flexibility:

.. code-block:: python

    def __init__(
        self,
        q_network: hk.TransformedWithState,
        optimizer: optax.GradientTransformation = None,
        ...
    ) -> None:

        if optimizer is None:
            optimizer = optax.adam(1e-3)

        self.init = jax.jit(partial(self.init, q_network=q_network, optimizer=optimizer, ...))

        ...

By implementing the constructor in this manner, users gain the flexibility to define their own architecture as follows:

.. code-block:: python

    @hk.transform_with_state
    def q_network(x: Array) -> Array:
        return hk.nets.MLP([64, 64, 2])(x)

    rl = RLib(
        agent_type=QLearning,
        agent_params={
            'q_network': q_network,
            'optimizer': optax.rmsprop(3e-4, decay=0.95, eps=1e-2)
        },
        ...
    )

During the development of a DRL agent, our library offers a set of :ref:`utility functions <JAX>` for your convenience.
Among these functions is gradient_step, designed to streamline parameter updates for the agent using JAX and optax.
In the following example code snippet, we showcase the implementation of a step function responsible for performing
a single step, taking into account the network, optimizer, and the implemented loss function:

.. code-block:: python

    from reinforced_lib.utils.jax_utils import gradient_step

    step_fn=partial(
        gradient_step,
        optimizer=optimizer,
        loss_fn=partial(self.loss_fn, q_network=q_network, ...)
    )

Our Python library also includes a pre-built :ref:`experience replay buffer <Experience Replay>`, which is commonly
utilized in DRL agents. The following code provides an illustrative example of how to use this utility:

.. code-block:: python

    from reinforced_lib.utils.experience_replay import experience_replay, ExperienceReplay, ReplayBuffer

    er = experience_replay(
        experience_replay_buffer_size,
        experience_replay_batch_size,
        obs_space_shape,
        act_space_shape
    )

    ...

    replay_buffer = er.init()

    ...

    replay_buffer = er.append(replay_buffer, prev_env_state, action, reward, terminal, env_state)
    perform_update = er.is_ready(replay_buffer)

    for _ in range(experience_replay_steps):
        batch = er.sample(replay_buffer, key)
        ...

Developing a DRL agent may pose challenges, so we strongly recommend thoroughly studying an example code of one of our
`DRL agents <https://github.com/m-wojnar/reinforced-lib/tree/main/reinforced_lib/agents/deep/>`_ prior to building
your custom agent.


Summary
-------

To sum everything up one more time:

1. All agents inherit from the ``BaseAgent`` class.
2. The agent's state is defined as a dataclass that inherits from the ``AgentState`` class.
3. The agent's behavior is determined by implementing the static methods ``init``, ``update``, and ``sample``.
4. Utilizing ``jax.jit`` can significantly increase the agent's performance.
5. Although not mandatory, it is highly recommended to implement the ``parameter_space``, ``update_observation_space``,
   and ``sample_observation_space`` properties.
6. Implementing a custom deep reinforcement learning agent is possible using the JAX ecosystem and utility functions
   provided by the library.