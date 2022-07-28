from collections import defaultdict
from typing import Any

from reinforced_lib.agents.base_agent import BaseAgent
from reinforced_lib.logs.base_logger import BaseLogger


class LogsObserver:
    def __init__(self) -> None:
        self._observation_loggers = defaultdict(list)
        self._agent_state_loggers = defaultdict(list)
        self._program_started = False

    def add_observation_logger(self, observation_name: str, logger: BaseLogger) -> None:
        self._observation_loggers[logger].append(observation_name)

    def add_agent_state_logger(self, state_attribute_name: str, logger: BaseLogger) -> None:
        self._agent_state_loggers[logger].append(state_attribute_name)

    def update_observations(self, observations: Any) -> None:
        pass

    def update_agent_state(self, agent_state: BaseAgent) -> None:
        pass
