import sys

import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')

import jax
import jax.numpy as jnp
import numpy as np
import inspect
from reinforced_lib.agents.mab import ThompsonSampling
from functools import wraps
import inspect
import chex
import jax.numpy as jnp
import jax.tree_util as tree

def flatten_args(tree_args_fun, treedef):
    @wraps(tree_args_fun)
    def flat_args_fun(*leaves):
        tree_args = tree.tree_unflatten(treedef, leaves)
        tree_ret = tree_args_fun(*tree_args)
        return tree.tree_leaves(tree_ret)

    return flat_args_fun

def make_converter(f,example):
    """

    Parameters
    ----------
    f a function
    example arguments in tree format
    """
    leaves,treedef = tree.tree_flatten(example)

    flat_fun = flatten_args(f,treedef)

    inputs = [[ (f'arg{i}',l) for i,l in enumerate(leaves)]]
    converter = tf.lite.TFLiteConverter.experimental_from_jax([flat_fun],inputs)
    return converter

def main():
    k = jax.random.PRNGKey(42)
    ts = ThompsonSampling(16)

    state = ts.init(k)
    new_state = ts.update(state,k,1,2,1,0.1)
    a = ts.sample(new_state,k,1.)

    converter = make_converter(ts.update,tree.tree_map(jnp.asarray,[state,k,1,2,1,0.1]))

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    tflite_update = converter.convert()
    with open('update.tflite', 'wb') as f:
        f.write(tflite_update)

    interpreter = tf.lite.Interpreter(model_content=tflite_update)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    # args = tree.tree_leaves([state,k,1,2,1,0.1])
    # args = tree.tree_map(jnp.asarray,args)
    #
    # for a,d in zip(args,input_details):
    #     interpreter.set_tensor(d['index'],a)

    interpreter.invoke()

    return

if __name__ == '__main__':

    main()

    ...