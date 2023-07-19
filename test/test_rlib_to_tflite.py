import os
import unittest
from glob import glob

import haiku as hk
import numpy as np
import optax

from reinforced_lib.agents.deep import DDPG
from reinforced_lib.agents.mab import ThompsonSampling
from reinforced_lib.exts import Gymnasium
from reinforced_lib.rlib import RLib


class TestRLibToTflite(unittest.TestCase):
    def test_sample_only_export(self):
        try:
            import tensorflow as tf
        except ModuleNotFoundError:
            return

        rl = RLib(
            agent_type=ThompsonSampling,
            agent_params={'n_arms': 4},
            no_ext_mode=True
        )
        rl.to_tflite(agent_id=0, sample_only=True)

        with open(glob(f'{os.path.expanduser("~")}/rlib-ThompsonSampling-0-*-init.tflite')[0], 'rb') as f:
            interpreter = tf.lite.Interpreter(model_content=f.read())
            interpreter.allocate_tensors()
            interpreter.invoke()
            key = [interpreter.get_tensor(od['index']) for od in interpreter.get_output_details()]

        with open(glob(f'{os.path.expanduser("~")}/rlib-ThompsonSampling-0-*-sample.tflite')[0], 'rb') as f:
            interpreter = tf.lite.Interpreter(model_content=f.read())
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            ins = key + [np.arange(4).astype(np.float32)]

            for d, a in zip(input_details, ins):
                interpreter.set_tensor(d['index'], a)

            interpreter.invoke()
            action, key = [interpreter.get_tensor(od['index']) for od in interpreter.get_output_details()]

        assert isinstance(action, (int, np.int32))
        assert isinstance(key, np.ndarray)
        assert key.shape == (2,)

    def test_full_export(self):
        try:
            import tensorflow as tf
        except ModuleNotFoundError:
            return

        rl = RLib(
            agent_type=ThompsonSampling,
            agent_params={'n_arms': 4},
            no_ext_mode=True
        )
        rl.to_tflite()

        with open(glob(f'{os.path.expanduser("~")}/rlib-ThompsonSampling-*-init.tflite')[0], 'rb') as f:
            interpreter = tf.lite.Interpreter(model_content=f.read())
            interpreter.allocate_tensors()
            interpreter.invoke()
            state = [interpreter.get_tensor(od['index']) for od in interpreter.get_output_details()]

        with open(glob(f'{os.path.expanduser("~")}/rlib-ThompsonSampling-*-update.tflite')[0], 'rb') as f:
            interpreter = tf.lite.Interpreter(model_content=f.read())
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            ins = state + [
                np.asarray(2, dtype=np.int32),
                np.asarray([1.], dtype=np.float32),
                np.asarray([1], dtype=np.int32),
                np.asarray([0], dtype=np.int32)
            ]

            for d, a in zip(input_details, ins):
                interpreter.set_tensor(d['index'], a)

            interpreter.invoke()
            state = [interpreter.get_tensor(od['index']) for od in interpreter.get_output_details()]

        with open(glob(f'{os.path.expanduser("~")}/rlib-ThompsonSampling-*-sample.tflite')[0], 'rb') as f:
            interpreter = tf.lite.Interpreter(model_content=f.read())
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            ins = state + [np.arange(4).astype(np.float32)]

            for d, a in zip(input_details, ins):
                interpreter.set_tensor(d['index'], a)

            interpreter.invoke()
            action, key, state_alpha, state_beta = [interpreter.get_tensor(od['index']) for od in interpreter.get_output_details()]

        assert isinstance(action, (int, np.int32))
        assert isinstance(key, np.ndarray)
        assert key.shape == (2,)
        assert isinstance(state_alpha, np.ndarray)
        assert state_alpha.shape == (4, 1)
        assert isinstance(state_beta, np.ndarray)
        assert state_beta.shape == (4, 1)

    def test_drl_sample_only_export(self):
        try:
            import tensorflow as tf
        except ModuleNotFoundError:
            return

        @hk.transform_with_state
        def q_network(s, a):
            return hk.nets.MLP([64, 1])(s)

        @hk.transform_with_state
        def a_network(s):
            return hk.nets.MLP([32, 32, 1])(s)

        rl = RLib(
            agent_type=DDPG,
            agent_params={
                'q_network': q_network,
                'a_network': a_network,
                'q_optimizer': optax.adam(2e-3),
                'a_optimizer': optax.adam(1e-3)
            },
            ext_type=Gymnasium,
            ext_params={'env_id': 'Pendulum-v1'}
        )
        rl.to_tflite(agent_id=0, sample_only=True)

    def test_drl_full_export(self):
        try:
            import tensorflow as tf
        except ModuleNotFoundError:
            return

        @hk.transform_with_state
        def q_network(s, a):
            return hk.nets.MLP([64, 1])(s)

        @hk.transform_with_state
        def a_network(s):
            return hk.nets.MLP([32, 32, 1])(s)

        rl = RLib(
            agent_type=DDPG,
            agent_params={
                'q_network': q_network,
                'a_network': a_network,
                'q_optimizer': optax.adam(2e-3),
                'a_optimizer': optax.adam(1e-3)
            },
            ext_type=Gymnasium,
            ext_params={'env_id': 'Pendulum-v1'}
        )
        rl.to_tflite()


if __name__ == '__main__':
    unittest.main()
