from typing import Any, Dict, List, Tuple, Union

import gym
import jax.random

from reinforced_lib.agents.base_agent import BaseAgent
from reinforced_lib.exts.base_ext import BaseExt
from reinforced_lib.utils.exceptions import *


class RLib:
    """
    Main class of the library. Exposes a simple and intuitive interface to use the library.

    Parameters
    ----------
    agent_type : type, optional
        Type of selected agent. Must inherit from the BaseAgent class.
    agent_params : dict, optional
        Parameters of selected agent.
    ext_type : type, optional
        Type of selected extension. Must inherit from the BaseExt class.
    ext_params : dict, optional
        Parameters of selected extension.
    log_type : type or list[type], optional
        Types of selected logging modules.
    log_params : dict or list[dict], optional
        Parameters of selected logging modules.
    no_ext_mode : bool, default=False
        Pass observations directly to the agent (don't use the Extensions module).
    """

    def __init__(
            self, *,
            agent_type: type = None,
            agent_params: Dict[str, Any] = None,
            ext_type: type = None,
            ext_params: Dict[str, Any] = None,
            log_type: Union[type, List[type]] = None,
            log_params: Union[Dict[str, Any], List[Dict[str, Any]]] = None,
            no_ext_mode: bool = False
    ) -> None:
        self._no_ext_mode = no_ext_mode

        self._agent = None
        self._agent_type = agent_type
        self._agent_params = agent_params
        self._agents_states = []
        self._agents_keys = []

        self._ext = None
        self._log = []

        if ext_type:
            self.set_ext(ext_type, ext_params)

        if agent_type:
            self.set_agent(agent_type, agent_params)

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
        agent_params : dict, optional
            Parameters of selected agent.
        """

        if len(self._agents_states) > 0:
            raise ForbiddenAgentChangeError()

        if not issubclass(agent_type, BaseAgent):
            raise IncorrectAgentTypeError(agent_type)

        self._agent_type = agent_type
        self._agent_params = agent_params

        if not self._no_ext_mode and self._ext:
            agent_params = self._ext.get_agent_params(agent_type, agent_type.parameters_space(), agent_params)
            self._agent = agent_type(**agent_params)
            self._ext.setup_transformations(self._agent.update_observation_space, self._agent.sample_observation_space)
        else:
            agent_params = agent_params if agent_params else {}
            self._agent = agent_type(**agent_params)

    def set_ext(self, ext_type: type, ext_params: Dict = None) -> None:
        """
        Initializes extension of type 'ext_type' with parameters 'ext_params'. The extension type must inherit from
        the BaseExt class. The extension type cannot be changed after the first agent instance is initialized.

        Parameters
        ----------
        ext_type : type
            Type of selected extension. Must inherit from the BaseExt class.
        ext_params : dict, optional
            Parameters of selected extension.
        """

        if self._no_ext_mode:
            raise ForbiddenExtensionSetError()

        if len(self._agents_states) > 0:
            raise ForbiddenExtensionChangeError()

        if not issubclass(ext_type, BaseExt):
            raise IncorrectExtensionTypeError(ext_type)

        ext_params = ext_params if ext_params else {}
        self._ext = ext_type(**ext_params)

        if self._agent:
            agent_params = self._ext.get_agent_params(
                self._agent_type,
                self._agent_type.init_observation_space(),
                self._agent_params
            )
            self._agent = self._agent_type(**agent_params)
            self._ext.setup_transformations(self._agent.update_observation_space, self._agent.sample_observation_space)

    def set_log(self, log_type: Union[type, List[type]], log_params: Union[Dict, List[Dict]] = None) -> None:
        """
        Initializes logging modules of types 'log_type' with parameters 'log_params'.

        Parameters
        ----------
        log_type : type or list[type]
            Types of selected logging modules.
        log_params : dict or list[dict], optional
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
        Returns observation space of selected extension or agent (if 'no_ext_mode' is enabled).

        Returns
        -------
        space : gym.spaces.Space
            Observation space of selected extension or agent (if 'no_ext_mode' is enabled).
        """

        if self._no_ext_mode:
            if not self._agent:
                raise NoAgentError()
            else:
                return gym.spaces.Dict({
                    'update_observation_space': self._agent.update_observation_space,
                    'sample_observation_space': self._agent.sample_observation_space
                })
        else:
            if not self._ext:
                raise NoExtensionError()
            else:
                return self._ext.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Returns action space of selected agent.

        Returns
        -------
        space : gym.spaces.Space
            Action space of selected agent.
        """

        if not self._agent:
            raise NoAgentError()

        return self._agent.action_space()

    def init(self, seed: int = 42) -> int:
        """
        Initializes new instance of the agent.

        Parameters
        ----------
        seed : int, default=42
            A number used to initialize the JAX pseudo-random number generator.

        Returns
        -------
        id : int
            The identifier of created instance.
        """

        agent_id = len(self._agents_states)
        init_key, key = jax.random.split(jax.random.PRNGKey(seed))

        self._agents_states.append(self._agent.init(init_key))
        self._agents_keys.append(key)

        return agent_id

    def sample(
            self,
            agent_id: int = 0,
            *args,
            update_observations: Union[Dict, Tuple, Any] = None,
            sample_observations: Union[Dict, Tuple, Any] = None,
            **kwargs
    ) -> Any:
        r"""
        Takes the extension state as input, updates the agent state, and returns the next action selected by 
        the agent. If 'no_ext_mode' is disabled, observations are passed by args and kwargs (observations must
        match selected extension observation space). If 'no_ext_mode' is enabled, observations must be passed 
        by 'update_observations' and 'sample_observations' parameters (observations must match agents 
        'update_observation_space' and 'sample_observation_space'). If there are no agent instance initialized,
        the method automatically initializes the first instance.

        Parameters
        ----------
        agent_id : int, default=0
            The identifier of agent instance.
        *args : tuple
            Extension observations.
        update_observations : dict or tuple or any, optional
            Observations used when 'no_ext_mode' is enabled (must match agents 'update_observation_space').
        sample_observations : dict or tuple or any, optional
            Observations used when 'no_ext_mode' is enabled (must match agents 'sample_observation_space').
        **kwargs : dict
            Extension observations.

        Returns
        -------
        action : Any
            Action selected by the agent.
        """

        if not self._agent:
            raise NoAgentError()

        if not self._no_ext_mode and not self._ext:
            raise NoExtensionError()

        if len(self._agents_states) == 0:
            self.init()

        key, update_key, sample_key = jax.random.split(self._agents_keys[agent_id], 3)
        state = self._agents_states[agent_id]

        if not self._no_ext_mode:
            update_observations, sample_observations = self._ext.transform(*args, **kwargs)

        if isinstance(update_observations, dict):
            state = self._agent.update(state, update_key, **update_observations)
        elif isinstance(update_observations, tuple):
            state = self._agent.update(state, update_key, *update_observations)
        else:
            state = self._agent.update(state, update_key, update_observations)

        if isinstance(sample_observations, dict):
            state, action = self._agent.sample(state, sample_key, **sample_observations)
        elif isinstance(sample_observations, tuple):
            state, action = self._agent.sample(state, sample_key, *sample_observations)
        else:
            state, action = self._agent(state, sample_key, sample_observations)

        self._agents_states[agent_id] = state
        self._agents_keys[agent_id] = key

        return action

    def fit(self, agent_id: int = 0) -> None:
        """
        Trains selected agent instance.

        Parameters
        ----------
        agent_id : int, default=0
            The identifier of agent instance.
        """

        if not self._agent:
            raise NoAgentError()

        if not self._no_ext_mode and not self._ext:
            raise NoExtensionError()

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
        agent_id : int, default=0
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
        id : int
            The identifier of loaded instance.
        """

        # TODO implement loading objects from file

        raise NotImplementedError()
