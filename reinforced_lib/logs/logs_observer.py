from collections import defaultdict
from typing import Any, Callable, Dict, List

import jax.numpy as jnp

from reinforced_lib.agents import BaseAgent
from reinforced_lib.logs import BaseLogger, Source, SourceType
from reinforced_lib.utils.exceptions import IncorrectLoggerTypeError, IncorrectSourceTypeError


class LogsObserver:
    def __init__(self) -> None:
        self._loggers_instances = {}
        self._loggers_sources = defaultdict(list)

        self._observations_loggers = defaultdict(list)
        self._agent_state_loggers = defaultdict(list)
        self._metrics_loggers = defaultdict(list)

    def add_logger(self, source: Source, logger_type: type, logger_params: Dict[str, Any]) -> None:
        if not issubclass(logger_type, BaseLogger):
            raise IncorrectLoggerTypeError(logger_type)

        if isinstance(source, tuple):
            if len(source) != 2 or not isinstance(source[0], str) or not hasattr(source[1], 'name'):
                raise IncorrectSourceTypeError(type(source))
        elif not isinstance(source, str):
            raise IncorrectSourceTypeError(type(source))

        logger = self._loggers_instances.get(logger_type, logger_type(**logger_params))

        if isinstance(source, tuple):
            if source[1] == SourceType.OBSERVATION:
                self._observations_loggers[logger].append((source, source[0]))
            elif source[1] == SourceType.STATE:
                self._agent_state_loggers[logger].append((source, source[0]))
            elif source[1] == SourceType.METRIC:
                self._metrics_loggers[logger].append((source, source[0]))
        elif isinstance(source, str):
            self._observations_loggers[logger].append((source, source))
            self._agent_state_loggers[logger].append((source, source))
            self._metrics_loggers[logger].append((source, source))

        self._loggers_sources[logger].append(source)
        self._loggers_instances[logger_type] = logger

    def init_loggers(self):
        for logger, sources in self._loggers_sources.items():
            logger.init(sources)

    def finish_loggers(self):
        for logger in self._loggers_sources.keys():
            logger.finish()

    def update_observations(self, observations: Any) -> None:
        if isinstance(observations, dict):
            self._update(self._observations_loggers, lambda name: observations.get(name, None))

    def update_agent_state(self, agent_state: BaseAgent) -> None:
        self._update(self._agent_state_loggers, lambda name: getattr(agent_state, name, None))

    def update_metrics(self, metric: Any, metric_name: str) -> None:
        self._update(self._metrics_loggers, lambda name: metric if name == metric_name else None)

    @staticmethod
    def _update(loggers: Dict[BaseLogger, List[str]], get_value: Callable) -> None:
        for logger, sources in loggers.items():
            for source, name in sources:
                if (value := get_value(name)) is not None:
                    if jnp.isscalar(value):
                        logger.log_scalar(source, value)
                    elif isinstance(value, dict):
                        logger.log_dict(source, value)
                    elif hasattr(value, '__len__'):
                        logger.log_array(source, value)
                    else:
                        logger.log_other(source, value)
