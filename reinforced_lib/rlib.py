from typing import Dict

from reinforced_lib.envs.base_env import BaseEnv
from reinforced_lib.agents.base_agent import BaseAgent
from reinforced_lib.utils.exceptions import *


class RLib:
    def __init__(
            self, *,
            agent_type: type = None,
            agent_params: Dict = None,
            env_type: type = None,
            env_params: Dict = None,
            log_type: type = None,
            log_params: Dict = None,
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
        log_type : type
            Type of selected logging module.
        log_params : Dict
            Parameters of selected logging module.
        no_env_mode : bool
            Pass observations directly to agent (don't use envs module).
        """

        self.agent = None
        self.env = None
        self.log = None

    def set_agent(self, agent_type: type, **agent_params) -> None:
        """

        Parameters
        ----------
        agent_type : type (inherited from BaseAgent)
            Type of selected agent.
        agent_params : Dict
            Parameters of selected agent.
        """

        self.agent = agent_type(**agent_params)

        if not isinstance(self.agent, BaseAgent):
            raise IncorrectAgentError(agent_type)

    def set_env(self, env_type: type, **env_params) -> None:
        """

        Parameters
        ----------
        env_type : type  (inherited from BaseEnv)
            Type of selected environment.
        env_params : Dict
            Parameters of selected environment.
        """

        if not self.agent:
            raise NoAgentError()

        self.env = env_type(self.agent.update_observation_space, self.agent.sample_observation_space, **env_params)

        if not isinstance(self.env, BaseEnv):
            raise IncorrectEnvironmentError(env_type)

    def set_log(self, log_type: type, **log_params) -> None:
        """

        Parameters
        ----------
        log_type : type
            Type of selected logging module.
        log_params
            Parameters of selected logging module.
        """

        raise NotImplementedError()
