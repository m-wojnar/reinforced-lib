import jax
import jax.numpy as jnp
import tensorflow as tf

from reinforced_lib.agents.mab import ThompsonSampling


def main2():
    k = jax.random.PRNGKey(42)
    ts = ThompsonSampling(16)

    state = ts.init(k)
    new_state = ts.update(state, k, 1, 2, 1, 0.1)
    a = ts.sample(new_state, k, 1.)

    arrays, defs = jax.tree_util.tree_flatten(state)

    def _update(*args):
        s = jax.tree_util.tree_unflatten(defs, args[:2])
        tmp = ts.update(s, *args[2:])
        arrs, _ = jax.tree_util.tree_flatten(tmp)
        x = args[0].at[1].set(4)

        return x

    converter = tf.lite.TFLiteConverter.experimental_from_jax([_update],
                                                              [[('alpha', state.alpha),
                                                                ('beta', state.beta),
                                                                ('k', k),
                                                                ('action', jnp.asarray(1)),
                                                                ('n_successful', jnp.asarray(1)),
                                                                ("n_failed", jnp.asarray(0)),
                                                                ("delta_time", jnp.asarray(0.1))]])
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

    args = jax.tree_util.tree_leaves(state) + jax.tree_util.tree_map(jnp.asarray, [k, 1, 2, 1, 0.1])

    # for a, d in zip(args, input_details):
    #     interpreter.set_tensor(d['index'], a)

    interpreter.invoke()
    return


def main3():
    print(f'JAX {jax.__version__}')
    print(f'tf {tf.__version__}')

    @jax.jit
    def _update(x):
        return x.at[0].set(4)

    converter = tf.lite.TFLiteConverter.experimental_from_jax([_update],
                                                              [[('x', jnp.ones(2))]])
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

    args = jnp.ones(2)

    expected = _update(args)
    print("Expected output:", expected)

    interpreter.set_tensor(input_details[0]['index'], args)
    interpreter.invoke()

    interpreter.set_tensor(input_details[0]['index'], args)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    print("Output:", output)

    return


def main4():
    print(f'JAX {jax.__version__}')
    print(f'tf {tf.__version__}')

    @jax.jit
    def _update(x):
        return x + 4 * jax.nn.one_hot(0, x.shape[0])

    converter = tf.lite.TFLiteConverter.experimental_from_jax([_update],
                                                              [[('x', jnp.ones(2))]])
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

    args = jnp.ones(2)

    expected = _update(args)

    interpreter.set_tensor(input_details[0]['index'], args)

    interpreter.invoke()
    assert jnp.allclose(interpreter.get_tensor(output_details[0]['index']), expected)
    return


if __name__ == '__main__':
    main2()
    main3()
    main4()
