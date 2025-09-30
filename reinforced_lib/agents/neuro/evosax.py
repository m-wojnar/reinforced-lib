from functools import partial
from typing import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass, Array, PRNGKey, Scalar, Shape
from evosax.algorithms.base import EvolutionaryAlgorithm, Params, State
from flax import linen as nn

from reinforced_lib.agents import BaseAgent, AgentState
from reinforced_lib.utils.jax_utils import forward


@dataclass
class EvosaxState(AgentState):
    r"""
    Container for the state of the evosax agent.

    Attributes
    ----------
    es_state : State
        The state of the evosax algorithm.
    population : dict
        The current population of the evolution strategy algorithm.
    best_params : Params
        The best parameters found so far.
    fitness : Array
        The fitness values of the current population.
    counter : int
        Number of the current step of the fitness evaluation.
    terminals : Array
        Whether the episodes have terminated.
    """

    es_state: State
    population: dict
    best_params: Params
    fitness: Array
    counter: int
    terminals: Array


class Evosax(BaseAgent):
    r"""
    Evolution strategies (ES)-based agent using the ``evosax`` library [12]_. This implementation maintains a population
    of candidate solutions (parameter vectors), evaluates them in parallel across environments, and updates the
    population by applying an evolutionary algorithm. Unlike gradient-based RL methods, this agent does not rely
    on backpropagation through the value or policy network. Instead, the network parameters are evolved using
    black-box optimization. This agent is suitable for environments with both discrete and continuous action spaces.

    **Note!** The user is responsible for providing appropriate network output in the correct format (e.g., discrete
    actions should be sampled from logits with ``jax.random.categorical`` inside the network definition).

    **Note!** This agent does not discount future rewards, therefore, the fitness is computed as a simple sum of
    rewards obtained during the evaluation phase.

    **Note!** This agent is compatible only with distribution-based evolution strategies from the ``evosax`` library
    (see `this list <https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/distribution_based>`_ for
    available algorithms). Population-based methods (`listed here <https://github.com/RobertTLange/evosax/tree/main/evosax/algorithms/population_based>`_
    will be supported in future releases.

    Parameters
    ----------
    network : nn.Module
        Architecture of the PPO agent network.
    evo_strategy : type
        Evolution strategy class from evosax.
    population_size : int
        Size of the population.
    obs_space_shape : Shape
        Shape of the observation space.
    act_space_shape : Shape, default=(1,)
        Shape of the action space. For discrete action spaces, use (1,).
    evo_strategy_kwargs : dict, default=None
        Parameters for the evolution strategy initialization. The population size and initial solution are set automatically.
    evo_strategy_default_params : dict, default=None
        Custom default parameters for the evolution strategy. If None, the default parameters are used.
    num_eval_steps : int, default=None
        Number of evaluation steps. If None, the evaluation runs until all episodes end.

    References
    ----------
    .. [12] Lange, R. (2022). evosax: JAX-based Evolution Strategies.
    """

    def __init__(
            self,
            network: nn.Module,
            evo_strategy: type,
            population_size: int,
            obs_space_shape: Shape,
            act_space_shape: Shape,
            evo_strategy_kwargs: dict = None,
            evo_strategy_default_params: dict = None,
            num_eval_steps: int = None
    ) -> None:

        if evo_strategy_kwargs is None:
            evo_strategy_kwargs = {}

        if evo_strategy_default_params is None:
            evo_strategy_default_params = {}

        self.obs_space_shape = obs_space_shape if jnp.ndim(obs_space_shape) > 0 else (obs_space_shape,)
        self.act_space_shape = act_space_shape if jnp.ndim(act_space_shape) > 0 else (act_space_shape,)
        self.population_size = population_size

        x_dummy = jnp.empty((1,) + self.obs_space_shape)
        variables = network.init(jax.random.key(0), x_dummy)
        num_params, params_format_fn = self.get_params_format_fn(variables)

        evo_strategy_kwargs['population_size'] = population_size
        evo_strategy_kwargs['solution'] = jnp.zeros(num_params)
        evo_strategy = evo_strategy(**evo_strategy_kwargs)

        self.init = jax.jit(partial(
            self.init,
            population_size=population_size,
            variables=variables,
            evo_strategy=evo_strategy,
            evo_strategy_default_params=evo_strategy_default_params
        ))
        self.update = jax.jit(partial(
            self.update,
            num_eval_steps=num_eval_steps,
            evo_strategy=evo_strategy
        ))
        self.sample = jax.jit(partial(
            self.sample,
            population_size=population_size,
            network=network,
            params_format_fn=params_format_fn
        ))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'obs_space_shape': gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), int)),
            'act_space_shape': gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), int)),
            'population_size': gym.spaces.Box(1, jnp.inf, (1,), int),
            'num_eval_steps': gym.spaces.Box(1, jnp.inf, (1,), int)
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_states': gym.spaces.Box(-jnp.inf, jnp.inf, (self.population_size,) + self.obs_space_shape, float),
            'actions': gym.spaces.Box(-jnp.inf, jnp.inf, (self.population_size,) + self.act_space_shape, float),
            'rewards': gym.spaces.Box(-jnp.inf, jnp.inf, (self.population_size,), float),
            'terminals': gym.spaces.MultiBinary(self.population_size)
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_states': gym.spaces.Box(-jnp.inf, jnp.inf, (self.population_size,) + self.obs_space_shape, float)
        })

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-jnp.inf, jnp.inf, (self.population_size,) + self.act_space_shape, float)

    @staticmethod
    def get_params_format_fn(init_params: dict) -> tuple[int, Callable]:
        # https://github.com/google/evojax/blob/main/evojax/util.py

        flat, tree = jax.tree.flatten(init_params)
        params_sizes = np.cumsum([np.prod(p.shape) for p in flat])

        def params_format_fn(params: Array) -> dict:
            params = jax.tree.map(
                lambda x, y: x.reshape(y.shape),
                jnp.split(params, params_sizes, axis=-1)[:-1],
                flat
            )
            return jax.tree.unflatten(tree, params)

        return params_sizes[-1], params_format_fn

    @staticmethod
    def init(
            key: PRNGKey,
            population_size: int,
            variables: dict,
            evo_strategy: EvolutionaryAlgorithm,
            evo_strategy_default_params: dict
    ) -> EvosaxState:
        r"""
        Initializes the evolution strategy state and the population. The fitness values, step counter,
        and terminals are initialized to zeros.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        population_size : int
            The size of the population.
        variables : dict
            The initialized parameters of the agent network.
        evo_strategy : EvolutionaryAlgorithm
            Initialized evosax evolution strategy.
        evo_strategy_default_params : dict
            Custom default parameters for the evolution strategy.

        Returns
        -------
        EvosaxState
            Initial state of the evosax agent.
        """

        es_params = evo_strategy.default_params
        es_params = es_params.replace(**evo_strategy_default_params)

        es_state = evo_strategy.init(key, variables, es_params)
        population, es_state = evo_strategy.ask(key, es_state, es_params)

        return EvosaxState(
            es_state=es_state,
            population=population,
            best_params=es_state.best_solution,
            fitness=jnp.zeros(population_size),
            counter=0,
            terminals=jnp.zeros(population_size, dtype=bool)
        )

    @staticmethod
    def update(
            state: EvosaxState,
            key: PRNGKey,
            env_states: Array,
            actions: Array,
            rewards: Scalar,
            terminals: bool,
            num_eval_steps: int,
            evo_strategy: EvolutionaryAlgorithm
    ) -> EvosaxState:
        r"""
        Updates the agent state after one evaluation step of the population. The method accumulates rewards
        into fitness values for each individual and tracks episode terminations. Once the evaluation is considered
        complete, either because all episodes have terminated (``num_eval_steps=None``) or because a fixed number
        of steps has been reached (``num_eval_steps`` specified), the population is evolved. The evolution strategy's
        ``tell`` method is called with the negative fitness values (as evosax minimizes the fitness), and a new population
        is generated using the ``ask`` method. The best parameters found so far are stored in the state.

        Parameters
        ----------
        state : EvosaxState
            The current state of the evosax agent.
        key : PRNGKey
            A PRNG key used as the random key.
        env_states : Array
            The current states of the environments.
        actions : Array
            The actions taken by the agent.
        rewards : Scalar
            The rewards received by the agent.
        terminals : bool
            Whether the episodes have terminated.
        num_eval_steps : int
            Number of evaluation steps. If None, the evaluation runs until all episodes end.
        evo_strategy : EvolutionaryAlgorithm
            The evosax evolution strategy.

        Returns
        -------
        EvosaxState
            The updated state of the evosax agent.
        """

        terminals = jnp.logical_or(state.terminals, terminals)

        if num_eval_steps is not None:
            fitness = state.fitness + rewards
            done = state.counter + 1 == num_eval_steps
        else:
            fitness = state.fitness + rewards * (~state.terminals)
            done = jnp.all(terminals)

        def do_evolve(state: EvosaxState, fitness: Array, terminals: Array, key: PRNGKey) -> EvosaxState:
            key_tell, key_ask = jax.random.split(key)

            es_params = evo_strategy.default_params
            es_state, _ = evo_strategy.tell(key_tell, state.population, -fitness, state.es_state, es_params)
            population, es_state = evo_strategy.ask(key_ask, es_state, es_params)

            return EvosaxState(
                es_state=es_state,
                population=population,
                best_params=es_state.best_solution,
                fitness=jnp.zeros_like(fitness),
                counter=0,
                terminals=jnp.zeros_like(terminals)
            )

        def carry_on(state: EvosaxState, fitness: Array, terminals: Array, _) -> EvosaxState:
            return state.replace(
                fitness=fitness,
                counter=state.counter + 1,
                terminals=terminals
            )

        return jax.lax.cond(
            done, do_evolve, carry_on,
            state, fitness, terminals, key
        )

    @staticmethod
    def sample(
            state: EvosaxState,
            key: PRNGKey,
            env_states: Array,
            population_size: int,
            network: nn.Module,
            params_format_fn: Callable
    ) -> Array:
        r"""
        Returns actions computed by the agents in the population. Note that the user is responsible for providing
        network output in the correct format.

        Parameters
        ----------
        state : EvosaxState
            The state of the PPO agent.
        key : PRNGKey
            A PRNG key used as the random key.
        env_states : Array
            The current state of the environment.
        population_size : int
            The size of the population.
        network : nn.Module
            The agent network.
        params_format_fn : Callable
            Function that formats the flattened parameters of the population members to the original neural network
            parameter format.

        Returns
        -------
        Array
            Selected actions.
        """

        @jax.vmap
        def vmap_sample(variables, key, env_states):
            variables = params_format_fn(variables)
            params, net_state = variables.pop('params'), variables
            outputs, _ = forward(network, params, net_state, key, env_states)
            return outputs

        keys = jax.random.split(key, population_size)
        return vmap_sample(state.population, keys, env_states)
