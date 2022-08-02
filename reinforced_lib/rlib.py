from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union

import os
import gym
import pickle
import lz4.frame
import jax.random

from reinforced_lib.agents import BaseAgent
from reinforced_lib.exts import BaseExt
from reinforced_lib.logs import Source
from reinforced_lib.logs.csv_logger import CsvLogger
from reinforced_lib.logs.logs_observer import LogsObserver
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
    loggers_type : type or list[type], optional
        Types of selected logging modules.
    loggers_sources : Source or list[Source], optional
        Names (and types) of selected sources.
    loggers_params : dict, optional
        Parameters of selected loggers.
    no_ext_mode : bool, default=False
        Pass observations directly to the agent (don't use the Extensions module).
    """

    def __init__(
            self, *,
            agent_type: type = None,
            agent_params: Dict[str, Any] = None,
            ext_type: type = None,
            ext_params: Dict[str, Any] = None,
            loggers_type: Union[type, List[type]] = None,
            loggers_sources: Union[Source, List[Source]] = None,
            loggers_params: Dict[str, Any] = None,
            no_ext_mode: bool = False
    ) -> None:
        self._ext = None
        self._no_ext_mode = no_ext_mode
        self._lz4_ext = ".pkl.lz4"

        self._agent = None
        self._agent_type = agent_type
        self._agent_params = agent_params
        self._agents_states = []
        self._agents_keys = []

        self._ext_type = ext_type
        self._ext_params = ext_params

        self._loggers_type = loggers_type
        self._loggers_sources = loggers_sources
        self._loggers_params = loggers_params
        self._logs_observer = LogsObserver()
        self._init_loggers = True

        if ext_type:
            self.set_ext(ext_type, ext_params)

        if agent_type:
            self.set_agent(agent_type, agent_params)

        if loggers_type and loggers_sources:
            self.set_loggers(loggers_type, loggers_sources, loggers_params)

    def __del__(self) -> None:
        """
        Automatically finalizes the library work.
        """

        self.finish()

    def finish(self) -> None:
        """
        Used to explicitly finalize the library work. In particular, it finishes loggers work.
        """

        self._logs_observer.finish_loggers()

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
                self._agent_type.parameters_space(),
                self._agent_params
            )
            self._agent = self._agent_type(**agent_params)
            self._ext.setup_transformations(self._agent.update_observation_space, self._agent.sample_observation_space)

    def set_loggers(
            self,
            loggers_types: Union[type, List[type]],
            loggers_sources: Union[Source, List[Source]],
            loggers_params: Dict[str, Any] = None
    ) -> None:
        """
        Initializes loggers that log environment observations, agents state or training metrics.
        'loggers_types' and 'loggers_sources' arguments can be objects of appropriate types or lists of object. 
        If user passes two objects or lists of the same lengths, function initializes modules with corresponding 
        types and names. If user passes one object (or list with only one object) and list of multiple objects, 
        function broadcasts passed objects. 'loggers_sources' items can be names of the logger sources
        (e.g. 'action') or tuples containing the name and the SourceType (e.g. ('action', SourceType.OBSERVATION)).
        If the name itself is inconclusive, behaviour depends on the implementation of the logger.

        Parameters
        ----------
        loggers_types : type or list[type]
            Types of selected logging modules.
        loggers_sources : Source or list[Source]
            Names of selected observations.
        loggers_params : dict, optional
            Parameters of selected logging modules.
        """

        if len(self._agents_states) > 0:
            raise ForbiddenLoggerSetError()

        loggers_params = loggers_params if loggers_params else {}
        loggers_types, loggers_sources = self._object_to_list(loggers_types), self._object_to_list(loggers_sources)
        loggers_types, loggers_sources = self._broadcast(loggers_types, loggers_sources)

        for logger_type, source in zip(loggers_types, loggers_sources):
            self._logs_observer.add_logger(source, logger_type, loggers_params)

    @staticmethod
    def _object_to_list(obj: Union[Any, List[Any]]) -> List[Any]:
        return obj if isinstance(obj, list) else [obj]

    @staticmethod
    def _broadcast(list_a: List[Any], list_b: List[Any]) -> Tuple[List[Any], List[Any]]:
        if len(list_a) == len(list_b):
            return list_a, list_b

        if len(list_a) == 1:
            return list_a * len(list_b), list_b

        if len(list_b) == 1:
            return list_a, list_b * len(list_a)

        raise TypeError('Incompatible length of given lists.')

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
        """Takes the extension state as input, updates the agent state, and returns the next action selected by
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

        if self._init_loggers:
            self._logs_observer.init_loggers()
            self._init_loggers = False

        if len(self._agents_states) == 0:
            self.init()

        key, update_key, sample_key = jax.random.split(self._agents_keys[agent_id], 3)
        state = self._agents_states[agent_id]

        if not self._no_ext_mode:
            update_observations, sample_observations = self._ext.transform(*args, **kwargs)

        if isinstance(update_observations, dict) and isinstance(sample_observations, dict):
            observations = {}
            observations.update(update_observations)
            observations.update(sample_observations)
            self._logs_observer.update_observations(observations)
        else:
            self._logs_observer.update_observations(update_observations)
            self._logs_observer.update_observations(sample_observations)

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

        self._logs_observer.update_agent_state(state)
        self._logs_observer.update_metrics(action, 'action')

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

        # TODO develop agents training

        raise NotImplementedError()

    def save(self, path: str = None) -> None:
        """
        Saves the state of the experiment to a file in lz4 format. For each agent both the state and the
        initialization parameters are saved. The environment extension is saved as well to fully reconstruct the
        experiment.

        Parameters
        ----------
        path : str
            Path to the checkpoint file. If ``.pkl.lz4`` extension not detected, it will be appended automaticly.
        """

        if path is None:
            path = os.path.join(os.path.expanduser("~"), f"checkpoint{self._lz4_ext}")
        elif path[-8:] != self._lz4_ext:
            path = path + self._lz4_ext

        experiment_state = {
            "agent_type": self._agent_type,
            "agent_params": self._agent_params,
            "agents": {
                agent_id: {
                    "state": self._agents_states[agent_id],
                    "key": self._agents_keys[0],
                } for agent_id in range(len(self._agents_states))
            },
            "ext_type": self._ext_type,
            "ext_params": self._ext_params,
            "loggers_type": self._loggers_type,
            "loggers_sources": self._loggers_sources,
            "loggers_params": self._loggers_params
        }

        with lz4.frame.open(path, 'wb') as f:
            f.write(pickle.dumps(experiment_state))
    
    @staticmethod
    def load(
        path: str,
        agent_params: Dict[str, Any] = None,
        ext_params: Dict[str, Any] = None,
    ) -> RLib:
        """
        Loads the state of the experiment from a file in lz4 format.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        agent_params : Dict[str, Any], optional
            _description_, by default None
        ext_params : Dict[str, Any], optional
            _description_, by default None
        """
        
        with lz4.frame.open(path, 'rb') as f:
            experiment_state = pickle.loads(f.read())
        
        rlib = RLib()
        
        rlib._agents_states = []
        rlib._agents_keys = []

        if ext_params:
            rlib.set_ext(experiment_state["ext_type"], ext_params)
        else:
            rlib.set_ext(experiment_state["ext_type"], experiment_state["ext_params"])
        
        if agent_params:
            rlib.set_agent(experiment_state["agent_type"], agent_params)
        else:
            rlib.set_agent(experiment_state["agent_type"], experiment_state["agent_params"])

        if experiment_state["loggers_type"]:
            rlib.set_loggers(
                experiment_state["loggers_type"],
                experiment_state["loggers_sources"],
                experiment_state["loggers_params"]
            )
        
        for agent_packed in experiment_state["agents"].values():
            rlib._agents_states.append(agent_packed["state"])
            rlib._agents_keys.append(agent_packed["key"])

        return rlib
