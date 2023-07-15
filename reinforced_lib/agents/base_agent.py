from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Tuple, Callable

import gymnasium as gym
import jax
from chex import dataclass, PRNGKey, ArrayTree

from reinforced_lib.utils.exceptions import UnimplementedSpaceError
from reinforced_lib.utils import is_array, is_dict


@dataclass
class AgentState:
    """
    Base class for agent state containers.
    """


class BaseAgent(ABC):
    """
    Base interface of agents.
    """

    @staticmethod
    @abstractmethod
    def init(key: PRNGKey, *args, **kwargs) -> AgentState:
        """
        Creates and initializes instance of the agent.
        """

        pass

    @staticmethod
    @abstractmethod
    def update(state: AgentState, key: PRNGKey, *args, **kwargs) -> AgentState:
        """
        Updates the state of the agent after performing some action and receiving a reward.
        """

        pass

    @staticmethod
    @abstractmethod
    def sample(state: AgentState, key: PRNGKey, *args, **kwargs) -> Any:
        """
        Selects the next action based on the current agent state.
        """

        pass

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        """
        Parameter space of the agent constructor in Gymnasium format.
        Type of returned value is required to be ``gym.spaces.Dict`` or ``None``.
        If ``None``, the user must provide all parameters manually.
        """

        return None

    @property
    def update_observation_space(self) -> gym.spaces.Space:
        """
        Observation space of the ``update`` method in Gymnasium format.
        If ``None``, the user must provide all parameters manually.
        """

        return None

    @property
    def sample_observation_space(self) -> gym.spaces.Space:
        """
        Observation space of the ``sample`` method in Gymnasium format.
        If ``None``, the user must provide all parameters manually.
        """

        return None

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Action space of the agent in Gymnasium format.
        """

        raise NotImplementedError()

    def export(self, init_key: PRNGKey, state: AgentState = None) -> Tuple[Any, Any, Any]:
        """
        Exports the agent to TensorFlow Lite format.

        Parameters
        ----------
        init_key : PRNGKey
            Key used to initialize the agent.
        state : AgentState, optional
            State of the agent to be exported. If not specified, the agent is initialized with ``init_key``.
        """

        import tensorflow as tf

        @dataclass
        class TfLiteState:
            state: ArrayTree
            key: PRNGKey

        def add_state(state: TfLiteState, args: Any) -> Any:
            if args is None:
                raise UnimplementedSpaceError()
            elif is_dict(args):
                return {**args, 'state': state}
            elif is_array(args):
                return [state] + list(args)
            else:
                return [state, args]

        def flatten_args(tree_args_fun: Callable, treedef: ArrayTree) -> Callable:
            @wraps(tree_args_fun)
            def flat_args_fun(*leaves):
                tree_args = jax.tree_util.tree_unflatten(treedef, leaves)

                if is_dict(tree_args):
                    tree_ret = tree_args_fun(**tree_args)
                else:
                    tree_ret = tree_args_fun(*tree_args)

                return jax.tree_util.tree_leaves(tree_ret)

            return flat_args_fun

        def make_converter(fun: Callable, arguments: Any) -> tf.lite.TFLiteConverter:
            leaves, treedef = jax.tree_util.tree_flatten(arguments)
            flat_fun = flatten_args(fun, treedef)

            inputs = [[(f'arg{i}', l) for i, l in enumerate(leaves)]]
            converter = tf.lite.TFLiteConverter.experimental_from_jax([flat_fun], inputs)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
                tf.lite.OpsSet.SELECT_TF_OPS     # enable TensorFlow ops.
            ]

            return converter

        def init() -> TfLiteState:
            return TfLiteState(
                state=self.init(init_key),
                key=init_key
            )

        def sample(state: TfLiteState, *args, **kwargs) -> Tuple[Any, TfLiteState]:
            sample_key, key = jax.random.split(state.key)
            action = self.sample(state.state, sample_key, *args, **kwargs)
            return action, TfLiteState(state=state.state, key=key)

        def update(state: TfLiteState, *args, **kwargs) -> TfLiteState:
            update_key, key = jax.random.split(state.key)
            new_state = self.update(state.state, update_key, *args, **kwargs)
            return TfLiteState(state=new_state, key=key)

        if state is None:
            state = init()
        else:
            state = TfLiteState(state=state, key=init_key)

        update_args = add_state(state, self.update_observation_space.sample())
        sample_args = add_state(state, self.sample_observation_space.sample())

        tfl_init = make_converter(init, []).convert()
        tfl_update = make_converter(update, update_args).convert()
        tfl_sample = make_converter(sample, sample_args).convert()

        return tfl_init, tfl_update, tfl_sample
