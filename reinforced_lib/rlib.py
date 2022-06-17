from typing import Dict, List, Tuple, Union

import jax.random

from reinforced_lib.agents.base_agent import BaseAgent
from reinforced_lib.envs.base_env import BaseEnv
from reinforced_lib.utils.exceptions import *


class RLib:
    """
    Main class of the library. Exposes a simple and intuitive interface to use the library.

    Parameters
    ----------
    agent_type : type
        Type of selected agent. Must inherit from the BaseAgent class.
    agent_params : Dict
        Parameters of selected agent.
    env_type : type
        Type of selected environment. Must inherit from the BaseEnv class.
    env_params : Dict
        Parameters of selected environment.
    log_type : Union[type, List[type]]
        Types of selected logging modules.
    log_params : Union[Dict, List[Dict]]
        Parameters of selected logging modules.
    no_env_mode : bool
        Pass observations directly to agent (don't use envs module).
    """

    def __init__(
            self, *,
            agent_type: type = None,
            agent_params: Dict = None,
            env_type: type = None,
            env_params: Dict = None,
            log_type: Union[type, List[type]] = None,
            log_params: Union[Dict, List[Dict]] = None,
            no_env_mode: bool = False
    ) -> None:
        self._no_env_mode = no_env_mode

        self._agent = None
        self._agents_states = []
        self._agents_keys = []

        self._env = None
        self._env_type = None
        self._env_params = None

        self._log = []

        if agent_type:
            self.set_agent(agent_type, agent_params)

        if env_type:
            self.set_env(env_type, env_params)

        if log_type:
            self.set_log(log_type, log_params)

    def set_agent(self, agent_type: type, agent_params: Dict = None) -> None:
        """
        Initializes agent of type 'agent_type' with parameters 'agent_params'. The agent type must inherit from
        the BaseAgent class. The agent type cannot be changed after the first agent instance is initialized.

        Parameters
        ----------
        agent_type : type
            Type of selected agent. Must inherit from the BaseAgent class.
        agent_params : Dict
            Parameters of selected agent.
        """

        if len(self._agents_states) > 0:
            raise ForbiddenAgentChangeError()

        if not issubclass(agent_type, BaseAgent):
            raise IncorrectAgentTypeError(agent_type)

        agent_params = agent_params if agent_params else {}
        self._agent = agent_type(**agent_params)

        if not self._no_env_mode and self._env:
            self.set_env(self._env_type, self._env_params)

    def set_env(self, env_type: type, env_params: Dict = None) -> None:
        """
        Initializes environment of type 'env_type' with parameters 'env_params'. The environment type must inherit from
        the BaseEnv class. The environment type cannot be changed after the first agent instance is initialized.

        Parameters
        ----------
        env_type : type
            Type of selected environment. Must inherit from the BaseEnv class.
        env_params : Dict
            Parameters of selected environment.
        """

        if self._no_env_mode:
            raise ForbiddenEnvironmentSetError()

        if not self._agent:
            raise NoAgentError()

        if len(self._agents_states) > 0:
            raise ForbiddenEnvironmentChangeError()

        if not issubclass(env_type, BaseEnv):
            raise IncorrectEnvironmentTypeError(env_type)

        env_params = env_params if env_params else {}
        self._env = env_type(self._agent.update_observation_space, self._agent.sample_observation_space, **env_params)

        self._env_type = env_type
        self._env_params = env_params

    def set_log(self, log_type: Union[type, List[type]], log_params: Union[Dict, List[Dict]] = None) -> None:
        """
        Initializes logging modules of types 'log_type' with parameters 'log_params'.

        Parameters
        ----------
        log_type : Union[type, List[type]]
            Types of selected logging modules.
        log_params : Union[Dict, List[Dict]]
            Parameters of selected logging modules.
        """

        if isinstance(log_type, list):
            log_params = log_params if log_params else [{} for _ in range(len(log_type))]
        else:
            log_params = log_params if log_params else {}

        # TODO create and implement logging modules initialization

        raise NotImplementedError()

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Returns observation space of selected environment or agent (if 'no_env_mode' is enabled).

        Returns
        -------
        out : gym.spaces.Space
            Observation space of selected environment or agent (if 'no_env_mode' is enabled).
        """

        if self._no_env_mode:
            if not self._agent:
                raise NoAgentError()
            else:
                return gym.spaces.Dict({
                    'update_observation_space': self._agent.update_observation_space,
                    'sample_observation_space': self._agent.sample_observation_space
                })
        else:
            if not self._env:
                raise NoEnvironmentError()
            else:
                return self._env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Returns action space of selected agent.

        Returns
        -------
        out : gym.spaces.Space
            Action space of selected agent.
        """

        if not self._agent:
            raise NoAgentError()

        return self._agent.action_space()

    def init(self, seed: int = None) -> int:
        """
        Initializes new instance of the agent.

        Parameters
        ----------
        seed : int
            A number used to initialize the JAX pseudo-random number generator.

        Returns
        -------
        out : int
            The identifier of created instance.
        """

        agent_id = len(self._agents_states)
        seed = seed if seed else 42

        self._agents_states.append(self._agent.init())
        self._agents_keys.append(jax.random.PRNGKey(seed))

        return agent_id

    def sample(
            self,
            agent_id: int = 0,
            *args,
            update_observations: Union[Dict, Tuple, Any] = None,
            sample_observations: Union[Dict, Tuple, Any] = None,
            **kwargs
    ) -> Any:
        """
        Takes the environment state as input, updates the agent state, and returns the next action selected by 
        the agent. If 'no_env_mode' is disabled, observations are passed by *args and **kwargs (observations must
        match selected environment observation space). If 'no_env_mode' is enabled, observations must be passed 
        by 'update_observations' and 'sample_observations' parameters (observations must match agents 
        'update_observation_space' and 'sample_observation_space'). If there are no agent instance initialized,
        the method automatically initializes the first instance.

        Parameters
        ----------
        agent_id : int
            The identifier of agent instance.
        args : Tuple
            Environment observations.
        update_observations : Union[Dict, Tuple, Any]
            Observations used when 'no_env_mode' is enabled (must match agents 'update_observation_space').
        sample_observations : Union[Dict, Tuple, Any]
            Observations used when 'no_env_mode' is enabled (must match agents 'sample_observation_space').
        kwargs : Dict
            Environment observations.

        Returns
        -------
        out : Any
            Action selected by the agent.
        """

        if not self._agent:
            raise NoAgentError()

        if not self._no_env_mode and not self._env:
            raise NoEnvironmentError()

        if len(self._agents_states) == 0:
            self.init()

        self._agents_keys[agent_id], key = jax.random.split(self._agents_keys[agent_id])
        state = self._agents_states[agent_id]

        if not self._no_env_mode:
            update_observations, sample_observations = self._env.transform(*args, **kwargs)

        if isinstance(update_observations, dict):
            state = self._agent.update(state, **update_observations)
        elif isinstance(update_observations, tuple):
            state = self._agent.update(state, *update_observations)
        else:
            state = self._agent.update(state, update_observations)

        if isinstance(sample_observations, dict):
            state, action = self._agent.sample(state, key, **sample_observations)
        elif isinstance(sample_observations, tuple):
            state, action = self._agent.sample(state, key, *sample_observations)
        else:
            state, action = self._agent(state, key, sample_observations)

        self._agents_states[agent_id] = state
        return action

    def fit(self, agent_id: int = 0) -> None:
        """
        Trains selected agent instance.

        Parameters
        ----------
        agent_id : int
            The identifier of agent instance.
        """

        if not self._agent:
            raise NoAgentError()

        if not self._no_env_mode and not self._env:
            raise NoEnvironmentError()

        if len(self._agents_states) == 0:
            self.init()

        state = self._agents_states[agent_id]

        # TODO develop agents training

        raise NotImplementedError()

    def save_agent_state(self, agent_id: int = 0, path: str = None) -> None:
        """
        Saves selected agent instance to file in X format.

        Parameters
        ----------
        agent_id : int
            The identifier of agent instance.
        path : str
            Path to the output file.
        """

        if not path:
            raise ValueError('No path is specified.')

        state = self._agents_states[agent_id]
        key = self._agents_keys[agent_id]

        obj_to_save = (state, key)

        # TODO implement objects saving and agree on a format

        raise NotImplementedError()

    def load_agent_state(self, path) -> int:
        """
        Loads agent instance from file in X format.

        Parameters
        ----------
        path : str
            Path to the input file.

        Returns
        -------
        out : int
            The identifier of loaded instance.
        """

        # TODO implement loading objects from file

        raise NotImplementedError()
