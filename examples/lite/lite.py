import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')

import jax
import jax.numpy as jnp
import numpy as np
import inspect
from reinforced_lib.agents.mab import ThompsonSampling


if __name__ == '__main__':




    # converter = tf.lite.TFLiteConverter.experimental_from_jax([lambda x: x+1],
    #                                                           [[('x',jnp.ones(()))]])
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    #     tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    # ]
    #
    # tflite_update = converter.convert()
    # with open('update.tflite', 'wb') as f:
    #     f.write(tflite_update)
    #
    # interpreter = tf.lite.Interpreter(model_content=tflite_update)
    # interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    #
    # args = jax.tree_util.tree_map(jnp.asarray,[2])
    #
    # for a,d in zip(args,input_details):
    #     interpreter.set_tensor(d['index'],a)
    #
    # interpreter.invoke()



    # rl = RLib(
    #     agent_type=ThompsonSampling,
    #     ext_type=IEEE_802_11_ax_RA
    # )
    k = jax.random.PRNGKey(42)
    ts = ThompsonSampling(16)

    state = ts.init(k)
    new_state = ts.update(state,k,1,2,1,0.1)
    a = ts.sample(new_state,k,1.)

    k = jax.random.PRNGKey(42)
    converter = tf.lite.TFLiteConverter.experimental_from_jax([ts.init], [[('k', k)]])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        #tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    tflite_init = converter.convert()

    arrays, defs = jax.tree_util.tree_flatten(state)

    #inspect.signature(ts.update)
    sig = inspect.signature(ts.update)

    def _update(*args):
        s = jax.tree_util.tree_unflatten(defs, args[:2])
        arrs,_ = jax.tree_util.tree_flatten(ts.update(s,*args[2:]))


    converter = tf.lite.TFLiteConverter.experimental_from_jax([_update],
                                                              [[('alpha', state.alpha),
                                                                ('beta', state.beta),
                                                                ('k', k),
                                                                ('action', jnp.asarray(1)),
                                                                ('n_successful',jnp.asarray(1)),
                                                                ("n_failed",jnp.asarray(0)),
                                                                ("delta_time",jnp.asarray(0.1))]])
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

    # working END ____________

    args = jax.tree_util.tree_leaves(state)+jax.tree_util.tree_map(jnp.asarray,[k,1,2,1,0.1])

    for a,d in zip(args,input_details):
        interpreter.set_tensor(d['index'],a)

    interpreter.invoke()


    converter = tf.lite.TFLiteConverter.experimental_from_jax([ts.sample],
                                                              [[('state', state), ('k', k), ('context', 1)]])

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    tflite_update = converter.convert()

    ...