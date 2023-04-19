from functools import wraps
from functools import wraps
from typing import Tuple, Any

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import tensorflow as tf
from chex import PRNGKey, ArrayTree

from reinforced_lib.agents import BaseAgent
from reinforced_lib.agents.mab import ThompsonSampling


def flatten_args(tree_args_fun, treedef):
    @wraps(tree_args_fun)
    def flat_args_fun(*leaves):
        tree_args = tree.tree_unflatten(treedef, leaves)
        tree_ret = tree_args_fun(*tree_args)
        return tree.tree_leaves(tree_ret)

    return flat_args_fun


def make_converter(f, example):
    """

    Parameters
    ----------
    f a function
    example arguments in tree format
    """
    leaves, treedef = tree.tree_flatten(example)

    flat_fun = flatten_args(f, treedef)

    inputs = [[(f'arg{i}', l) for i, l in enumerate(leaves)]]
    converter = tf.lite.TFLiteConverter.experimental_from_jax([flat_fun], inputs)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    return converter


@chex.dataclass
class TfLiteState:
    agent_state: ArrayTree
    key: PRNGKey


def export(agent: BaseAgent, base_name: str, init_args, update_args, sample_args):
    def init(*args, **kwargs) -> TfLiteState:
        return TfLiteState(
            agent_state=agent.init(*args, **kwargs),
            key=jax.random.PRNGKey(42)
        )

    def sample(state: TfLiteState, *args, **kwargs) -> Tuple[Any, TfLiteState]:
        sample_key, key = jax.random.split(state.key)
        action = agent.sample(state.agent_state, sample_key, *args, **kwargs)
        return action, TfLiteState(agent_state=state.agent_state, key=key)

    def update(state: TfLiteState, *args, **kwargs) -> TfLiteState:
        update_key, key = jax.random.split(state.key)
        new_state = agent.update(state.agent_state, update_key, *args, **kwargs)
        return TfLiteState(agent_state=new_state, key=key)

    s0 = init(*init_args)
    converter = make_converter(init, init_args)
    tfl_init = converter.convert()
    with open(f'{base_name}_init.tflite', 'wb') as f:
        f.write(tfl_init)

    args = [s0] + update_args
    s1 = update(*args)
    converter = make_converter(update, args)
    tfl_update = converter.convert()

    with open(f'{base_name}_update.tflite', 'wb') as f:
        f.write(tfl_update)

    args = [s1] + sample_args
    converter = make_converter(sample, args)
    tfl_sample = converter.convert()
    with open(f'{base_name}_sample.tflite', 'wb') as f:
        f.write(tfl_sample)

    return


def main():
    k = jax.random.PRNGKey(42)
    ts = ThompsonSampling(16)

    export(ts, 'ts', init_args=[k], update_args=[1, 2, 1, 0.1], sample_args=[1.])

    with open('ts_init.tflite', 'rb') as f:
        interpreter = tf.lite.Interpreter(model_content=f.read())
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], k)
        interpreter.invoke()
        outs = [interpreter.get_tensor(od['index']) for od in interpreter.get_output_details()]

    with open('ts_update.tflite', 'rb') as f:
        interpreter = tf.lite.Interpreter(model_content=f.read())
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        # types
        ins = outs + tree.tree_map(jnp.asarray,[1, 2, 1, 0.1])
        for d, a in zip(input_details, ins):
            interpreter.set_tensor(d['index'], a)
        interpreter.invoke()
        next_outs = [interpreter.get_tensor(od['index']) for od in interpreter.get_output_details()]

    with open('ts_sample.tflite', 'rb') as f:
        interpreter = tf.lite.Interpreter(model_content=f.read())
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        ins = next_outs + [jnp.asarray(1.)]
        for d, a in zip(input_details, ins):
            interpreter.set_tensor(d['index'], a)
        interpreter.invoke()
        action_and_state = [interpreter.get_tensor(od['index']) for od in interpreter.get_output_details()]

    action = action_and_state[0]
    next_state = action_and_state[1:]


if __name__ == '__main__':
    main()

    ...
