.. _custom_agents:

Custom agents
=============

Although our library provides a palette of already implemented :ref:`agents <Agents>`, you might want to
add a personalised one to the collection. This guide is to help you with this task.


Customizing agents
------------------

To fully benefit from Reinforced-lib features, including JAX jit optimization, your agent
should inherit from the :ref:`abstract class <BaseAgent>` ``BaseAgent``. We present adding a
custom agent on a simple example of the epsilon-greedy agent:

.. code-block:: python

    class EGreedy(BaseAgent)

Firstly, we need to define the state of our agent, which in our case will hold

    * constant experiment rate (e),
    * quality values of each arm (Q),
    * number of each arms' tries (N),

and will inherit from AgentState:

.. code-block:: python
    
    @dataclass
    class EGreedyState(AgentState):

        Q: Array
        N: Array

Next, we can define the Epsilon-greedy agent, which will have 3 static methods:

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
    
    # This method updates the agents state after performing some action and receiving a reward
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
    ) -> Tuple[EGreedyState, jnp.int32]:

        # We further want to jax.jit this function, so basic 'if' is not allowed here
        return jax.lax.cond(

            # Split PRNGkey to use it twice
            epsilon_key, choice_key = jax.random.split(key)

            # The agent experiments with probability e
            jax.random.uniform(epsilon_key) < e,

            # On exploration, agent chooses a random arm
            lambda: (state, jax.random.choice(choice_key, state.Q.size)),

            # On exploitation, agent chooses the best known arm
            lambda: (state, jnp.argmax(state.Q))
        )

Having defined these static methods, we can define the class constructor:

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

        # Here, we can use the jax.jit() functionality with the previously
        # defined behaviour functions, to make the agent super fast
        self.init = jax.jit(partial(self.init, n_arms=self.n_arms))
        self.update = jax.jit(partial(self.update))
        self.sample = jax.jit(partial(self.sample, e=e))

Lastly, we must specify the parameter spaces that each of the implemented methods take.
This enables the library to automatically infer the necessary parameters from the environment.

.. code-block:: python

    # Parameters required by the agent constructor in OpenAI Gym format.
    # Type of returned value is required to be gym.spaces.Dict.
    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'e': gym.spaces.Box(0.0, 1.0, (1,), jnp.float32)
        })
    
    # Parameters required by the 'update' method in OpenAI Gym format.
    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'action': gym.spaces.Discrete(self.n_arms),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), jnp.float32)
        })
    
    # Parameters required by the 'sample' method in OpenAI Gym format.
    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})
    
    # Action returned by the agent in OpenAI Gym format.
    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(self.n_arms)

Now we have a ready to operate epsilon-greedy agent!


Template Agent
--------------

Here is all of the above code in one piece. You can copy-paste it and use as an inspiration
to create your own agent.

.. code-block:: python

    from functools import partial
    from typing import Tuple

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
            self.update = jax.jit(partial(self.update))
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
                n_arms: jnp.int32,
                optimistic_start: Scalar
        ) -> EGreedyState:

            return EGreedyState(
                Q=(optimistic_start * jnp.ones(n_arms)),
                N=jnp.ones(n_arms, dtype=jnp.int32)
            )

        @staticmethod
        def update(
            state: EGreedyState,
            key: PRNGKey,
            action: jnp.int32,
            reward: Scalar,
            alpha: Scalar
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
        ) -> Tuple[EGreedyState, jnp.int32]:

            epsilon_key, choice_key = jax.random.split(key)

            return jax.lax.cond(
                jax.random.uniform(epsilon_key) < e,
                lambda: (state, jax.random.choice(choice_key, state.Q.size)),
                lambda: (state, jnp.argmax(state.Q))
            )



Sum up
------

To sum everything up one more time:

1. Custom agent inherits from the `BaseAgent`.
2. We implement the abstract methods *init()*, *update()* and *sample()*.
3. We use *jax.jit()* to optimize the agent's performance.
4. We provide the required parameters in the format of *OpenAI Gym* spaces.

The built-in implementation of the epsilon-greedy agent with an addition of optional optimistic start and an exponential
recency-weighted average update can be found
`here <https://github.com/m-wojnar/reinforced-lib/blob/main/reinforced_lib/agents/mab/e_greedy.py>`_.
