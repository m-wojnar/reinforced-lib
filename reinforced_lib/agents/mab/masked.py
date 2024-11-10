from functools import partial

import chex
import gym
import jax
import jax.numpy as jnp
from chex import dataclass, PRNGKey

from reinforced_lib.agents import BaseAgent, AgentState


@dataclass
class MaskedState:
    agent_state:AgentState
    mask: chex.Array


class Masked(BaseAgent):
    r"""
    Meta agent supporting dynamic change of number of arms.

    **This agent is highly experimental and is expected to be used with an extreme caution.**
    In particular, this agent makes the following strong assumptions:

    - Each entry in the base agent state has the first dimension corresponding to an arm.
    - The base agent must be stochastic as this agent uses rejection sampling to choose a possible action

    Parameters
    ----------

    agent: BaseAgent
        A base agnet whose action are masked
    mask: chex.Array
        Binary mask array of the length equal to the number of arms. positive entry mars the masked action
    """
    agent: BaseAgent

    def __init__(self, agent:BaseAgent, mask:chex.Array):

        self.agent = agent

        self.init = jax.jit(partial(self.init, agent = agent, mask = mask))
        self.update = jax.jit(partial(self.update, agent = agent))
        self.sample = jax.jit(partial(self.sample, agent = agent))


    @staticmethod
    def init(key: PRNGKey, agent:BaseAgent, mask:chex.Array, *args, **kwargs) -> AgentState:
        return MaskedState(agent_state=agent.init(key, *args, **kwargs), mask=mask)

    @staticmethod
    def update(state: AgentState, key: PRNGKey, *args, **kwargs,) -> AgentState:
        agent = kwargs.pop('agent')
        agent_state=agent.update(state.agent_state,key, *args, **kwargs)
        tree_mask = jax.tree_util.tree_map(lambda _: jnp.expand_dims(state.mask, 1),state.agent_state )

        agent_state = jax.tree_util.tree_map(lambda s,ns,m: jnp.where(m,s, ns),
                                             state.agent_state,
                                             agent_state,
                                             tree_mask)


        return MaskedState(agent_state=agent_state, mask=state.mask)

    @staticmethod
    def sample(state: AgentState, key: PRNGKey, *args, **kwargs) -> any:
        agent = kwargs.pop('agent')
        k1,k2=jax.random.split(key,2)
        action = agent.sample(state.agent_state, k1, *args, **kwargs)

        def body_fun(x:tuple):
            a,k = x
            k1,k2 = jax.random.split(k,2)
            k2 = jax.random.fold_in(k2,a)
            action = agent.sample(state.agent_state, k1, *args, **kwargs)
            return action, k2

        action,_ = jax.lax.while_loop(cond_fun=lambda a:state.mask[a[0]],
                                      body_fun=body_fun, init_val=(action, k2))
        return action



    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return self.agent.update_observation_space

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return self.agent.sample_observation_space

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.agent.action_space

    @staticmethod
    def parameter_space(cls) -> gym.spaces.Dict:
        
        cls.parameter_space()
