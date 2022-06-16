from typing import Dict, List, Union, Tuple

import jax.random

from reinforced_lib.agents.base_agent import BaseAgent
from reinforced_lib.envs.base_env import BaseEnv
from reinforced_lib.utils.exceptions import *


class RLib:
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
        """

        Parameters
        ----------
        agent_type : type (inherited from BaseAgent)
            Type of selected agent.
        agent_params : Dict
            Parameters of selected agent.
        env_type : type  (inherited from BaseEnv)
            Type of selected environment.
        env_params : Dict
            Parameters of selected environment.
        log_type : Union[type, List[type]]
            Types of selected logging modules.
        log_params : Union[Dict, List[Dict]]
            Parameters of selected logging modules.
        no_env_mode : bool
            Pass observations directly to agent (don't use envs module).
        """

        self._no_env_mode = no_env_mode

        self._agent = None
        self._agents_states = []
        self._agents_keys = []

        self._env = None
        self._log = []

        if agent_type:
            self.set_agent(agent_type, agent_params)

        if env_type:
            self.set_env(env_type, env_params)

        if log_type:
            self.set_log(log_type, log_params)

    def set_agent(self, agent_type: type, agent_params: Dict = None) -> None:
        """


        Parameters
        ----------
        agent_type : type (inherited from BaseAgent)
            Type of selected agent.
        agent_params : Dict
            Parameters of selected agent.
        """

        if len(self._agents_states) > 0:
            raise ForbiddenAgentChangeError()

        if not issubclass(agent_type, BaseAgent):
            raise IncorrectAgentTypeError(agent_type)

        agent_params = agent_params if agent_params else {}
        self._agent = agent_type(**agent_params)

    def set_env(self, env_type: type, env_params: Dict = None) -> None:
        """

        Parameters
        ----------
        env_type : type  (inherited from BaseEnv)
            Type of selected environment.
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

    def set_log(self, log_type: Union[type, List[type]], log_params: Union[Dict, List[Dict]] = None) -> None:
        """

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

        raise NotImplementedError()

    def observation_space(self) -> gym.spaces.Space:
        """

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

    def action_space(self) -> gym.spaces.Space:
        """

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

        Parameters
        ----------
        seed : int
            Number used to initialize the JAX pseudo-random number generator.

        Returns
        -------
        out : int
            Identifier of created agent.
        """

        agent_id = len(self._agents_states)
        seed = seed if seed else 0

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

        Parameters
        ----------
        agent_id : int
            The identifier of agent (returned by 'init' function).
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
            state = self._agent(state, **update_observations)
        elif isinstance(update_observations, tuple):
            state = self._agent(state, *update_observations)
        else:
            state = self._agent(state, update_observations)

        if isinstance(sample_observations, dict):
            state, action = self._agent(state, key, **sample_observations)
        elif isinstance(sample_observations, tuple):
            state, action = self._agent(state, key, *sample_observations)
        else:
            state, action = self._agent(state, key, sample_observations)

        self._agents_states[agent_id] = state
        return action

    def fit(self, agent_id: int = 0):
        """

        Parameters
        ----------
        agent_id : int
            The identifier of agent (returned by 'init' function).
        """

        if not self._agent:
            raise NoAgentError()

        if not self._no_env_mode and not self._env:
            raise NoEnvironmentError()

        if len(self._agents_states) == 0:
            self.init()

        state = self._agents_states[agent_id]

        raise NotImplementedError()
