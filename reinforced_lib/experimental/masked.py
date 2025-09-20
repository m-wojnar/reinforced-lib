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
    def init(key: PRNGKey, *args: tuple, agent: BaseAgent, mask: Array, **kwargs: dict) -> MaskedState:
        r"""
        Initialize the masked agent state given the mask and the base agent.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        args : tuple
            Positional arguments passed to the base agent init method.
        agent : BaseAgent
            A base agent whose state is initialized.
        mask : Array
            Binary mask array of the length equal to the number of arms.
        kwargs : dict
            Keyword arguments passed to the base agent init method.

        Returns
        -------
        MaskedState
            Initialized masked agent state.
        """

        return MaskedState(agent_state=agent.init(key, *args, **kwargs), mask=mask)

    @staticmethod
    def update(state: MaskedState, key: PRNGKey, *args: tuple, agent: BaseAgent, **kwargs: dict) -> MaskedState:
        r"""
        Update the base agent state. The entries corresponding to the masked actions are not updated.

        Parameters
        ----------
        state : MaskedState
            Current masked agent state.
        key : PRNGKey
            A PRNG key used as the random key.
        args : tuple
            Positional arguments passed to the base agent update method.
        agent : BaseAgent
            A base agent whose state is updated.
        kwargs : dict
            Keyword arguments passed to the base agent update method.

        Returns
        -------
        MaskedState
            Updated masked agent state.
        """

        tree_mask = jax.tree.map(lambda _: jnp.expand_dims(state.mask, 1), state.agent_state)
        agent_state = agent.update(state.agent_state, key, *args, **kwargs)
        agent_state = jax.tree.map(lambda s, ns, m: jnp.where(m, s, ns), state.agent_state, agent_state, tree_mask)
        return MaskedState(agent_state=agent_state, mask=state.mask)

    @staticmethod
    def sample(state: MaskedState, key: PRNGKey, *args, agent: BaseAgent, **kwargs) -> int:
        r"""
        Sample an action from the base agent. If the sampled action is masked, resample until an unmasked action is found.

        Parameters
        ----------
        state : MaskedState
            Current masked agent state.
        key : PRNGKey
            A PRNG key used as the random key.
        args : tuple
            Positional arguments passed to the base agent sample method.
        agent : BaseAgent
            A base agent whose action is sampled.
        kwargs : dict
            Keyword arguments passed to the base agent sample method.

        Returns
        -------
        int
            Sampled action.
        """

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
