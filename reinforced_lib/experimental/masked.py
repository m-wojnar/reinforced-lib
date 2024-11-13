from functools import partial

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, dataclass

from reinforced_lib.agents import AgentState, BaseAgent


@dataclass
class MaskedState(AgentState):
    agent_state: AgentState
    mask: Array


class Masked(BaseAgent):
    r"""
    Meta agent supporting dynamic change of number of arms.

    **This agent is highly experimental and is expected to be used with an extreme caution.**
    In particular, this agent makes the following strong assumptions:

    - Each entry in the base agent state has the first dimension corresponding to an arm.
    - The base agent must be stochastic as this agent uses rejection sampling to choose a possible action

    Example usage of the agent can be found in the test `test/experimental/test_masked.py`.

    Parameters
    ----------
    agent : BaseAgent
        A MAB agent type which actions are masked.
    mask : Array
        Binary mask array of the length equal to the number of arms. Positive entries are the masked actions.
    """

    def __init__(self, agent: BaseAgent, mask: Array) -> None:
        self.init = jax.jit(partial(self.init, agent=agent, mask=mask))
        self.update = jax.jit(partial(self.update, agent=agent))
        self.sample = jax.jit(partial(self.sample, agent=agent))

    @staticmethod
    def init(key: PRNGKey, *args, agent: BaseAgent, mask: Array, **kwargs) -> MaskedState:
        return MaskedState(agent_state=agent.init(key, *args, **kwargs), mask=mask)

    @staticmethod
    def update(state: MaskedState, key: PRNGKey, *args, agent: BaseAgent, **kwargs) -> MaskedState:
        tree_mask = jax.tree.map(lambda _: jnp.expand_dims(state.mask, 1), state.agent_state)
        agent_state = agent.update(state.agent_state, key, *args, **kwargs)
        agent_state = jax.tree.map(lambda s, ns, m: jnp.where(m, s, ns), state.agent_state, agent_state, tree_mask)
        return MaskedState(agent_state=agent_state, mask=state.mask)

    @staticmethod
    def sample(state: MaskedState, key: PRNGKey, *args, agent: BaseAgent, **kwargs) -> int:
        sample_key, while_key = jax.random.split(key, 2)
        action = agent.sample(state.agent_state, sample_key, *args, **kwargs)

        def cond_fn(carry: tuple) -> bool:
            action, _ = carry
            return state.mask[action]

        def body_fn(carry: tuple) -> tuple:
            action, key = carry
            sample_key, key = jax.random.split(key)
            key = jax.random.fold_in(key, action)
            action = agent.sample(state.agent_state, sample_key, *args, **kwargs)
            return action, key

        action, _ = jax.lax.while_loop(cond_fn, body_fn, (action, while_key))
        return action
