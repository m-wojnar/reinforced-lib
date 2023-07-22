from __future__ import annotations

import os
import pickle
from typing import Union

import cloudpickle
import gymnasium as gym
import jax.random
import lz4.frame
from chex import dataclass

from reinforced_lib.agents import BaseAgent
from reinforced_lib.exts import BaseExt
from reinforced_lib.logs import Source
from reinforced_lib.logs.logs_observer import LogsObserver
from reinforced_lib.utils.exceptions import *
from reinforced_lib.utils import is_scalar, timestamp


@dataclass
class AgentContainer:
    """
    Class containing the state of a given agent and all its dependencies.

    Attributes
    ----------
    state : BaseAgent
        Current state of the agent.
    key : jax.random.PRNGKey
        A PRNG key used as the random key.
    action : any
        Action selected by the agent.
    step : int
        Current step of the agent.
    """

    state: BaseAgent
    key: jax.random.PRNGKey
    action: any
    step: int


class RLib:
    """
    Main class of the library. Exposes a simple and intuitive interface to use the library.

    Parameters
    ----------
    agent_type : type, optional
        Type of the selected agent. Must inherit from the ``BaseAgent`` class.
    agent_params : dict, optional
        Parameters of the selected agent.
    ext_type : type, optional
        Type of the selected extension. Must inherit from the ``BaseExt`` class.
    ext_params : dict, optional
        Parameters of the selected extension.
    logger_types : type or list[type], optional
        Types of the selected logging modules. Must inherit from the ``BaseLogger`` class.
    logger_sources : Source or list[Source], optional
        Sources to log.
    logger_params : dict, optional
        Parameters of the selected loggers.
    no_ext_mode : bool, default=False
        Pass observations directly to the agent (do not use the extensions).
    auto_checkpoint : int, optional
        Automatically save the experiment every ``auto_checkpoint`` steps.
        If ``None``, the automatic checkpointing is disabled.
    auto_checkpoint_path : str, optional, default=~
        Path to the directory where the automatic checkpoints will be saved.
    """

    def __init__(
            self, *,
            agent_type: type = None,
            agent_params: dict[str, any] = None,
            ext_type: type = None,
            ext_params: dict[str, any] = None,
            logger_types: Union[type, list[type]] = None,
            logger_sources: Union[Source, list[Source]] = None,
            logger_params: dict[str, any] = None,
            no_ext_mode: bool = False,
            auto_checkpoint: int = None,
            auto_checkpoint_path: str = None
    ) -> None:
        self._lz4_ext = ".pkl.lz4"
        self._default_path = os.path.expanduser("~")
        self._auto_checkpoint = auto_checkpoint
        self._auto_checkpoint_path = auto_checkpoint_path if auto_checkpoint_path else self._default_path

        self._agent = None
        self._agent_type = agent_type
        self._agent_params = agent_params
        self._agent_containers = []

        self._ext = None
        self._no_ext_mode = no_ext_mode
        self._ext_type = ext_type
        self._ext_params = ext_params

        self._logger_types = logger_types
        self._logger_sources = logger_sources
        self._logger_params = logger_params
        self._logs_observer = LogsObserver()
        self._init_loggers = True
        self._cumulative_reward = 0.0

        if ext_type:
            self.set_ext(ext_type, ext_params)

        if agent_type:
            self.set_agent(agent_type, agent_params)

        if logger_types:
            self.set_loggers(logger_types, logger_sources, logger_params)

    def __del__(self) -> None:
        """
        Automatically finalizes the library work.
        """

        self.finish()

    def finish(self) -> None:
        """
        Used to explicitly finalize the library's work. In particular, it finishes the logger's work.
        """

        self._logs_observer.finish_loggers()

    def set_agent(self, agent_type: type, agent_params: dict = None) -> None:
        """
        Initializes an agent of type ``agent_type`` with parameters ``agent_params``. The agent type must inherit from
        the ``BaseAgent`` class. The agent type cannot be changed after the first agent instance has been initialized.

        Parameters
        ----------
        agent_type : type
            Type of the selected agent. Must inherit from the ``BaseAgent`` class.
        agent_params : dict, optional
            Parameters of the selected agent.
        """

        if len(self._agent_containers) > 0:
            raise ForbiddenAgentChangeError()

        if not issubclass(agent_type, BaseAgent):
            raise IncorrectAgentTypeError(agent_type)

        self._agent_type = agent_type
        self._agent_params = agent_params

        if not self._no_ext_mode and self._ext:
            agent_params = self._ext.get_agent_params(agent_type, agent_type.parameter_space(), agent_params)
            self._agent = agent_type(**agent_params)
            self._ext.setup_transformations(self._agent.update_observation_space, self._agent.sample_observation_space)
        else:
            agent_params = agent_params if agent_params else {}
            self._agent = agent_type(**agent_params)

    def set_ext(self, ext_type: type, ext_params: dict = None) -> None:
        """
        Initializes an extension of type ``ext_type`` with parameters ``ext_params``. The extension type must inherit
        from the ``BaseExt`` class. The extension type cannot be changed after the first agent instance has been
        initialized.

        Parameters
        ----------
        ext_type : type
            Type of selected extension. Must inherit from the ``BaseExt`` class.
        ext_params : dict, optional
            Parameters of the selected extension.
        """

        if self._no_ext_mode:
            raise ForbiddenExtensionSetError()

        if len(self._agent_containers) > 0:
            raise ForbiddenExtensionChangeError()

        if not issubclass(ext_type, BaseExt):
            raise IncorrectExtensionTypeError(ext_type)

        self._ext_type = ext_type
        self._ext_params = ext_params

        ext_params = ext_params if ext_params else {}
        self._ext = ext_type(**ext_params)

        if self._agent:
            agent_params = self._ext.get_agent_params(
                self._agent_type,
                self._agent_type.parameter_space(),
                self._agent_params
            )
            self._agent = self._agent_type(**agent_params)
            self._ext.setup_transformations(self._agent.update_observation_space, self._agent.sample_observation_space)

    def set_loggers(
            self,
            logger_types: Union[type, list[type]],
            logger_sources: Union[Source, list[Source]] = None,
            logger_params: dict[str, any] = None
    ) -> None:
        """
        Initializes loggers of types ``logger_types`` with parameters ``logger_params``. The logger types must inherit
        from the ``BaseLogger`` class. The logger types cannot be changed after the first agent instance has been
        initialized. ``logger_types`` and ``logger_sources`` can be objects or lists of objects, the function broadcasts
        them to the same length. The ``logger_sources`` parameter specifies the sources to log. A source can be None
        (then the logger is used to log a custom values passed by the ``log`` method), a name of the sources (e.g.,
        "action") or tuple containing the name and the ``SourceType`` (e.g., ``("action", SourceType.OBSERVATION)``).
        If the name itself is inconclusive (e.g., it occurs as a metric and as an observation), the behaviour depends
        on the implementation of the logger.

        Parameters
        ----------
        logger_types : type or list[type]
            Types of the selected logging modules.
        logger_sources : Source or list[Source], optional
            Sources to log.
        logger_params : dict, optional
            Parameters of the selected logging modules.
        """

        if not self._init_loggers:
            raise ForbiddenLoggerSetError()

        self._logger_types = logger_types
        self._logger_sources = logger_sources
        self._logger_params = logger_params

        logger_params = logger_params if logger_params else {}
        logger_types, logger_sources = self._object_to_list(logger_types), self._object_to_list(logger_sources)
        logger_types, logger_sources = self._broadcast(logger_types, logger_sources)

        for logger_type, source in zip(logger_types, logger_sources):
            self._logs_observer.add_logger(source, logger_type, logger_params)

    @staticmethod
    def _object_to_list(obj: Union[any, list[any]]) -> list[any]:
        return obj if isinstance(obj, list) else [obj]

    @staticmethod
    def _broadcast(list_a: list[any], list_b: list[any]) -> tuple[list[any], list[any]]:
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
        Returns the observation space of the selected extension (or agent, if ``no_ext_mode`` is set).

        Returns
        -------
        gym.spaces.Space
            Observation space of the selected extension or agent.
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
        Returns the action space of the selected agent.

        Returns
        -------
        gym.spaces.Space
            Action space of the selected agent.
        """

        if not self._agent:
            raise NoAgentError()

        return self._agent.action_space

    def init(self, seed: int = 42) -> int:
        """
        Initializes a new instance of the agent.

        Parameters
        ----------
        seed : int, default=42
            Number used to initialize the JAX pseudo-random number generator.

        Returns
        -------
        int
            Identifier of the created instance.
        """

        agent_id = len(self._agent_containers)
        init_key, key = jax.random.split(jax.random.PRNGKey(seed))

        self._agent_containers.append(AgentContainer(
            state=self._agent.init(init_key),
            key=key,
            action=None,
            step=0
        ))

        return agent_id

    def sample(
            self,
            *args,
            agent_id: int = 0,
            is_training: bool = True,
            update_observations: Union[dict, tuple, any] = None,
            sample_observations: Union[dict, tuple, any] = None,
            **kwargs
    ) -> any:
        """
        Takes the extension state as an input, updates the agent state, and returns the next action selected by
        the agent. If ``no_ext_mode`` is disabled, observations are passed by args and kwargs (the observations must
        match the extension observation space). If ``no_ext_mode`` is enabled, observations must be passed
        by the ``update_observations`` and ``sample_observations`` parameters (the observations must match the agent's
        ``update_observation_space`` and ``sample_observation_space``). If there are no agent instances initialized,
        the method automatically initializes the first instance. If the ``is_training`` flag is set, the ``update`` and
        ``sample`` agent methods will be called. Otherwise, only the ``sample`` method will be called.

        Parameters
        ----------
        *args : tuple
            Environment observations.
        agent_id : int, default=0
            The identifier of the agent instance.
        is_training : bool
            Flag indicating whether the agent state should be updated in this step.
        update_observations : dict or tuple or any, optional
            Observations used when ``no_ext_mode`` is enabled (must match agent's ``update_observation_space``).
        sample_observations : dict or tuple or any, optional
            Observations used when ``no_ext_mode`` is enabled (must match agent's ``sample_observation_space``).
        **kwargs : dict
            Environment observations.

        Returns
        -------
        any
            Action selected by the agent.
        """

        if not self._agent:
            raise NoAgentError()

        if not self._no_ext_mode and not self._ext:
            raise NoExtensionError()

        update_observations = update_observations if update_observations else {}
        sample_observations = sample_observations if sample_observations else {}

        if self._init_loggers:
            self._logs_observer.init_loggers()
            self._init_loggers = False

        if len(self._agent_containers) == 0:
            self.init()

        key, update_key, sample_key = jax.random.split(self._agent_containers[agent_id].key, 3)
        state = self._agent_containers[agent_id].state
        action = self._agent_containers[agent_id].action
        step = self._agent_containers[agent_id].step

        if not self._no_ext_mode:
            update_observations, sample_observations = self._ext.transform(*args, action=action, **kwargs)

        all_observations = kwargs
        if isinstance(update_observations, dict) and isinstance(sample_observations, dict):
            all_observations |= update_observations
            all_observations |= sample_observations
            self._logs_observer.update_observations(all_observations)
        else:
            self._logs_observer.update_observations(update_observations)
            self._logs_observer.update_observations(sample_observations)

        if is_training and step > 0:
            if isinstance(update_observations, dict):
                state = self._agent.update(state, update_key, **update_observations)
            elif isinstance(update_observations, tuple):
                state = self._agent.update(state, update_key, *update_observations)
            else:
                state = self._agent.update(state, update_key, update_observations)

            if self._auto_checkpoint is not None and (step + 1) % self._auto_checkpoint == 0:
                checkpoint_path = os.path.join(self._auto_checkpoint_path, f'rlib-checkpoint-agent-{agent_id}-step-{step + 1}')
                self.save(checkpoint_path, agent_ids=agent_id)

        if isinstance(sample_observations, dict):
            action = self._agent.sample(state, sample_key, **sample_observations)
        elif isinstance(sample_observations, tuple):
            action = self._agent.sample(state, sample_key, *sample_observations)
        else:
            action = self._agent.sample(state, sample_key, sample_observations)

        self._logs_observer.update_agent_state(state)
        self._logs_observer.update_metrics(action, 'action')

        def log_reward(reward: float) -> None:
            self._cumulative_reward += reward
            self._logs_observer.update_metrics(reward, 'reward')
            self._logs_observer.update_metrics(self._cumulative_reward, 'cumulative')

        if 'reward' in all_observations:
            log_reward(all_observations['reward'])
        elif self._ext:
            try:
                if hasattr(self._ext, 'reward'):
                    log_reward(self._ext.reward(**all_observations))
                elif 'reward' in self._ext._observation_functions:
                    log_reward(self._ext._observation_functions['reward'](**all_observations))
            except TypeError:
                pass

        self._agent_containers[agent_id] = AgentContainer(
            state=state,
            key=key,
            action=action,
            step=step + 1
        )

        return action

    def save(self, path: str = None, *, agent_ids: Union[int, list[int]] = None) -> str:
        """
        Saves the state of the experiment to a file in lz4 format. For each agent, both the state and the initialization
        parameters are saved. The extension and loggers settings are saved as well to fully reconstruct the experiment.

        Parameters
        ----------
        path : str, optional
            Path to the checkpoint file. If none specified, saves to the default path.
            If the ``.pkl.lz4`` suffix is not detected, it will be appended automatically.
        agent_ids : int or array_like, optional
            The identifier of the agent instance(s) to save. If none specified, saves the state of all agents.
        
        Returns
        -------
        str
            Path to the saved checkpoint file.
        """

        if agent_ids is None:
            agent_ids = list(range(len(self._agent_containers)))
        elif is_scalar(agent_ids):
            agent_ids = [agent_ids]

        agent_containers = [self._agent_containers[agent_id] for agent_id in agent_ids]

        if path is None:
            path = os.path.join(self._default_path, f"rlib-checkpoint-{timestamp()}.pkl.lz4")
        elif path[-8:] != self._lz4_ext:
            path = path + self._lz4_ext

        experiment_state = {
            "agent_type": self._agent_type,
            "agent_params": self._agent_params,
            "agents": {
                agent_id: {
                    "state": agent.state,
                    "key": agent.key,
                    "action": agent.action,
                    "step": agent.step
                } for agent_id, agent in zip(agent_ids, agent_containers)
            },
            "ext_type": self._ext_type,
            "ext_params": self._ext_params,
            "logger_types": self._logger_types,
            "logger_sources": self._logger_sources,
            "logger_params": self._logger_params,
            "auto_checkpoint": self._auto_checkpoint
        }

        with lz4.frame.open(path, 'wb') as f:
            f.write(cloudpickle.dumps(experiment_state))

        return path

    @staticmethod
    def load(
            path: str,
            *,
            agent_params: dict[str, any] = None,
            ext_params: dict[str, any] = None,
            logger_types: Union[type, list[type]] = None,
            logger_sources: Union[Source, list[Source]] = None,
            logger_params: dict[str, any] = None
    ) -> RLib:
        """
        Loads the state of the experiment from a file in lz4 format.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        agent_params : dict[str, any], optional
            Dictionary of altered agent parameters with their new values, by default None.
        ext_params : dict[str, any], optional
            Dictionary of altered extension parameters with their new values, by default None.
        logger_types : type or list[type], optional
            Types of the selected logging modules. Must inherit from the ``BaseLogger`` class.
        logger_sources : Source or list[Source], optional
            Sources to log.
        logger_params : dict, optional
            Parameters of the selected loggers.
        """

        with lz4.frame.open(path, 'rb') as f:
            experiment_state = pickle.loads(f.read())

        rlib = RLib(
            auto_checkpoint=experiment_state["auto_checkpoint"],
            no_ext_mode=experiment_state["ext_type"] is None
        )

        rlib._agent_containers = []

        if experiment_state["ext_type"]:
            if ext_params:
                rlib.set_ext(experiment_state["ext_type"], ext_params)
            else:
                rlib.set_ext(experiment_state["ext_type"], experiment_state["ext_params"])

        if experiment_state["agent_type"]:
            if agent_params:
                rlib.set_agent(experiment_state["agent_type"], agent_params)
            else:
                rlib.set_agent(experiment_state["agent_type"], experiment_state["agent_params"])

        if logger_types:
            rlib.set_loggers(logger_types, logger_sources, logger_params)
        elif experiment_state["logger_types"]:
            rlib.set_loggers(
                experiment_state["logger_types"],
                experiment_state["logger_sources"],
                logger_params if logger_params else experiment_state["logger_params"]
            )

        for agent_id, agent_container in experiment_state["agents"].items():
            while agent_id >= len(rlib._agent_containers):
                rlib.init()

            rlib._agent_containers[agent_id] = AgentContainer(
                state=agent_container["state"],
                key=agent_container["key"],
                action=agent_container["action"],
                step=agent_container["step"]
            )

        return rlib

    def log(self, name: str, value: any) -> None:
        """
        Logs a custom value.

        Parameters
        ----------
        name : str
            The name of the value to log.
        value : any
            The value to log.
        """

        self._logs_observer.update_custom(value, name)

    def to_tflite(self, path: str = None, *, agent_id: int = None, sample_only: bool = False) -> None:
        """
        Converts the agent to a TensorFlow Lite model and saves it to a file.

        Parameters
        ----------
        path : str, optional
            Path to the output file.
        agent_id : int, optional
            The identifier of the agent instance to convert. If specified,
            state of the selected agent will be saved.
        sample_only : bool
            Flag indicating if the method should save only the sample function.
        """

        if not self._agent:
            raise NoAgentError()

        if len(self._agent_containers) == 0:
            self.init()

        if sample_only and agent_id is None:
            raise ValueError("Agent ID must be specified when saving sample function only.")

        if path is None:
            path = self._default_path

        if agent_id is None:
            init_tfl, update_tfl, sample_tfl = self._agent.export(
                init_key=jax.random.PRNGKey(42)
            )
        else:
            init_tfl, update_tfl, sample_tfl = self._agent.export(
                init_key=self._agent_containers[agent_id].key,
                state=self._agent_containers[agent_id].state,
                sample_only=sample_only
            )

        base_name = self._agent.__class__.__name__
        base_name += f'-{agent_id}-' if agent_id is not None else '-'
        base_name += timestamp()

        with open(os.path.join(path, f'rlib-{base_name}-init.tflite'), 'wb') as f:
            f.write(init_tfl)

        with open(os.path.join(path, f'rlib-{base_name}-sample.tflite'), 'wb') as f:
            f.write(sample_tfl)

        if not sample_only:
            with open(os.path.join(path, f'rlib-{base_name}-update.tflite'), 'wb') as f:
                f.write(update_tfl)
