.. _agents:

Agents
======

This module is a set of RL agents. You can either choose one of our built-in agents or implement
your agent with the help of this short guide.

.. _custom_agents:

Custom agents
-------------

To fully benefit from the reinforced-lib features, including the JAX jit optimization, your agent
should inherit from the :ref:`abstract class <base_agent>` ``BaseAgent``. We will present adding
custom agent on a simple example of epsilon-greedy agent:

.. code-block:: python

    class EGreedy(BaseAgent)

Firstly, we need to define a state of our agent, which in our case will hold

    * constant experiment rate (e),
    * quality values of each arm (Q),
    * number of each arms tries (N),

and will inherit from AgentState:

.. code-block:: python
    
    @chex.dataclass
    class EGreedyState(AgentState):

        e: jnp.float32
        Q: chex.Array
        N: chex.Array

Next, we can define the Epsilon-greedy agent, which will have 3 static methods:

.. code-block:: python
    
    # This method initializes the agent with 'n_arms' arms 
    @staticmethod
    def init(
        n_arms: jnp.int32, 
        e: jnp.float32
    ) -> EGreedyState:

        return EGreedyState(

            # The experiment rate e
            e=e,

            # The initial Q values are set as zeros, due to the lack of prior knowledge
            Q=jnp.zeros(n_arms),

            # The numbers of tries are set as ones, to avoid null division in Q value update
            N=jnp.ones(n_arms, dtype=jnp.int32)
        )
    
    # This method updates the agents state after performing some action and receiving a reward
    @staticmethod
    def update(
        state: EGreedyState,
        key: chex.PRNGKey,
        action: jnp.int32,
        reward: jnp.float32,
    ) -> EGreedyState:

        return EGreedyState(

            # We do not change the experiment rate
            e=state.e,

            # Q value update
            Q=state.Q.at[action].add((1.0 / state.N[action]) * (reward - state.Q[action])),

            # Incrementing the number of tries on appropriate arm
            N=state.N.at[action].add(1)
        )
    
    # This method samples new action according to the agents state (experience)
    @staticmethod
    def sample(
        state: EGreedyState,
        key: chex.PRNGKey
    ) -> Tuple[EGreedyState, jnp.int32]:

        # We further want to jax.jit this function, so basic 'if' is not allowed here
        return jax.lax.cond(

            # The agent experiments with probability e
            jax.random.uniform(key) < state.e,

            # On exploration, agent chooses a random arm
            lambda: (state, jax.random.choice(key, state.Q.size)),

            # On exploitation, agent chooses the best known arm
            lambda: (state, jnp.argmax(state.Q))
        )

Having defined those static methods, we can implement the class constructor:

.. code-block:: python
    
    def __init__(
        self, 
        n_arms: jnp.int32, 
        e: jnp.float32
    ) -> None:

        # We specify the features of our agent
        self.n_arms = n_arms
        self.e = e

        # Here, we can use the jax.jit() functionality with the previously
        # defined behaviour functions, to make the agent super fast
        self.init = jax.jit(partial(self.init, n_arms=self.n_arms, e=self.e))
        self.update = jax.jit(partial(self.update))
        self.sample = jax.jit(partial(self.sample))

Lastly, we must specify the parameters spaces that each of the implemented method takes.
It will help the library to automatically infer the necessary parameters from the environment.

.. code-block:: python

    # Parameters required by the agents constructor in OpenAI Gym format. 
    # Type of returned value is required to be gym.spaces.Dict.
    @staticmethod
    def parameters_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'e': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32),
            'optimistic_start': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32)
        })
    
    # Parameters required by the 'update' method in OpenAI Gym format.
    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_arms),
            'reward': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32)
        })
    
    # Parameters required by the 'sample' method in OpenAI Gym format.
    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})
    
    # Action returned by the agent in OpenAI Gym format.
    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(self.n_arms)

Now we have a ready to operate epsilon-greedy agent! Putting the code together:

.. code-block:: python

    from functools import partial
    from typing import Tuple

    import chex
    import gym.spaces
    import jax
    import jax.numpy as jnp

    from reinforced_lib.agents.base_agent import BaseAgent, AgentState


    @chex.dataclass
    class EGreedyState(AgentState):

        e: jnp.float32
        Q: chex.Array
        N: chex.Array
    

    class EGreedy(BaseAgent):

        def __init__(
            self, 
            n_arms: jnp.int32, 
            e: jnp.float32
        ) -> None:

            self.n_arms = n_arms
            self.e = e

            self.init = jax.jit(partial(self.init, n_arms=self.n_arms, e=self.e))
            self.update = jax.jit(partial(self.update))
            self.sample = jax.jit(partial(self.sample))
        
        @staticmethod
        def parameters_space() -> gym.spaces.Dict:
            return gym.spaces.Dict({
                'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
                'e': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32),
                'optimistic_start': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32)
            })
        
        @property
        def update_observation_space(self) -> gym.spaces.Dict:
            return gym.spaces.Dict({
                'action': gym.spaces.Discrete(self.n_arms),
                'reward': gym.spaces.Box(0.0, jnp.inf, (1,), jnp.float32)
            })

        @property
        def sample_observation_space(self) -> gym.spaces.Dict:
            return gym.spaces.Dict({})

        @property
        def action_space(self) -> gym.spaces.Space:
            return gym.spaces.Discrete(self.n_arms)
 
        @staticmethod
        def init(
            n_arms: jnp.int32, 
            e: jnp.float32
        ) -> EGreedyState:

            return EGreedyState(
                e=e,
                Q=jnp.zeros(n_arms),
                N=jnp.ones(n_arms, dtype=jnp.int32)
            )
        
        @staticmethod
        def update(
            state: EGreedyState,
            key: chex.PRNGKey,
            action: jnp.int32,
            reward: jnp.float32,
        ) -> EGreedyState:

            return EGreedyState(
                e=state.e,
                Q=state.Q.at[action].add((1.0 / state.N[action]) * (reward - state.Q[action]))
                N=state.N.at[action].add(1)
            )

        @staticmethod
        def sample(
            state: EGreedyState,
            key: chex.PRNGKey
        ) -> Tuple[EGreedyState, jnp.int32]:

            return jax.lax.cond(
                jax.random.uniform(key) < state.e,
                lambda: (state, jax.random.choice(key, state.Q.size)),
                lambda: (state, jnp.argmax(state.Q))
            )

To sum up everything one more time:

1. Custom agent inherits from the `BaseAgent``
2. We implement the abstract methods *init()*, *update()* and *sample()*
3. We use *jax.jit()* to optimize the agents performance
4. We provide the required parameters in format of *OpenAI Gym* spaces

The built-in implementation of the epsilon-greedy agent, with addition of optional optimistic start,
can be found :ref:`here <e-greedy_agent>`.
        


.. _base_agent:

BaseAgent
---------

.. currentmodule:: reinforced_lib.agents.base_agent

.. autoclass:: AgentState
    :members:

.. autoclass:: BaseAgent
    :members:

List of agents
--------------

Thompson Sampling
~~~~~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.agents.thompson_sampling

.. autoclass:: ThompsonSamplingState
    :show-inheritance:
    :members:

.. autoclass:: ThompsonSampling
    :show-inheritance:
    :members:

.. _particle-filter_agent:

Particle Filter
~~~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.agents.wifi.particle_filter

.. autoclass:: ParticleFilterState
    :show-inheritance:
    :members:

.. autoclass:: ParticleFilter
    :show-inheritance:
    :members:

.. _e-greedy_agent:

Epsilon-greedy
~~~~~~~~~~~~~~

.. currentmodule:: reinforced_lib.agents.e_greedy

.. autoclass:: EGreedyState
    :show-inheritance:
    :members:

.. autoclass:: EGreedy
    :show-inheritance:
    :members:
