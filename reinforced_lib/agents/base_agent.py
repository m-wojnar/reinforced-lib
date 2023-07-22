from abc import ABC, abstractmethod
from functools import wraps, partial
from typing import Callable

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
    def sample(state: AgentState, key: PRNGKey, *args, **kwargs) -> any:
        """
        Selects the next action based on the current environment and agent state.
        """

        pass

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        """
        Parameters of the agent constructor in Gymnasium format. Type of returned value is required  to
        be ``gym.spaces.Dict`` or ``None``. If ``None``, the user must provide all parameters manually.
        """

        return None

    @property
    def update_observation_space(self) -> gym.spaces.Space:
        """
        Observation space of the ``update`` method in Gymnasium format. Allows to infer missing
        observations using an extensions and easily export the agent to TensorFlow Lite format.
        If ``None``, the user must provide all parameters manually.
        """

        return None

    @property
    def sample_observation_space(self) -> gym.spaces.Space:
        """
        Observation space of the ``sample`` method in Gymnasium format. Allows to infer missing
        observations using an extensions and easily export the agent to TensorFlow Lite format.
        If ``None``, the user must provide all parameters manually.
        """

        return None

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Action space of the agent in Gymnasium format.
        """

        raise NotImplementedError()

    def export(self, init_key: PRNGKey, state: AgentState = None, sample_only: bool = False) -> tuple[any, any, any]:
        """
        Exports the agent to TensorFlow Lite format.

        Parameters
        ----------
        init_key : PRNGKey
            Key used to initialize the agent.
        state : AgentState, optional
            State of the agent to be exported. If not specified, the agent is initialized with ``init_key``.
        sample_only : bool, optional
            If ``True``, the exported agent will only be able to sample actions, but not update its state.
        """

        try:
            import tensorflow as tf
        except ModuleNotFoundError:
            raise ModuleNotFoundError('TensorFlow installation is required to export agents to TensorFlow Lite.')

        @dataclass
        class TfLiteState:
            state: ArrayTree
            key: PRNGKey

        def append_value(value: any, value_name: str, args: any) -> any:
            if args is None:
                raise UnimplementedSpaceError()
            elif is_dict(args):
                return {value_name: value} | args
            elif is_array(args):
                return [value] + list(args)
            else:
                return [value, args]

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

        def make_converter(fun: Callable, arguments: any) -> tf.lite.TFLiteConverter:
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

        def sample(state: TfLiteState, *args, **kwargs) -> tuple[any, TfLiteState]:
            sample_key, key = jax.random.split(state.key)
            action = self.sample(state.state, sample_key, *args, **kwargs)
            return action, TfLiteState(state=state.state, key=key)

        def update(state: TfLiteState, *args, **kwargs) -> TfLiteState:
            update_key, key = jax.random.split(state.key)
            new_state = self.update(state.state, update_key, *args, **kwargs)
            return TfLiteState(state=new_state, key=key)

        def get_key() -> PRNGKey:
            return init_key

        def sample_without_state(state: AgentState, key: PRNGKey, *args, **kwargs) -> tuple[any, PRNGKey]:
            sample_key, key = jax.random.split(key)
            action = self.sample(state, sample_key, *args, **kwargs)
            return action, key

        if not sample_only:
            if state is None:
                state = init()
            else:
                state = TfLiteState(state=state, key=init_key)

            update_args = append_value(state, 'state', self.update_observation_space.sample())
            sample_args = append_value(state, 'state', self.sample_observation_space.sample())

            init_tfl = make_converter(init, []).convert()
            update_tfl = make_converter(update, update_args).convert()
            sample_tfl = make_converter(sample, sample_args).convert()

            return init_tfl, update_tfl, sample_tfl
        elif state is not None:
            sample_args = append_value(init_key, 'key', self.sample_observation_space.sample())

            init_tfl = make_converter(get_key, []).convert()
            sample_tfl = make_converter(partial(sample_without_state, state), sample_args).convert()

            return init_tfl, None, sample_tfl
        else:
            raise ValueError('Either `state` must be provided or `sample_only` must be False.')
