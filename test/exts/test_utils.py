import unittest

import gym.spaces

from reinforced_lib.exts.utils import *


class ObservationDecoratorTest(unittest.TestCase):
    def test_observation_info_attribute(self):
        @observation()
        def fn():
            return 0.0

        self.assertTrue(hasattr(fn, 'observation_info'))

    def test_observation_info_default_name(self):
        @observation()
        def fn():
            return 0.0

        self.assertEqual(fn.observation_info.name, 'fn')

    def test_observation_info_custom_name(self):
        @observation(observation_name='test_name')
        def fn():
            return 0.0

        self.assertEqual(fn.observation_info.name, 'test_name')

    def test_observation_info_type(self):
        @observation(observation_type=gym.spaces.Box(-1.0, 1.0))
        def fn():
            return 0.0

        self.assertEqual(fn.observation_info.type, gym.spaces.Box(-1.0, 1.0))


class ParameterDecoratorTest(unittest.TestCase):
    def test_parameter_info_attribute(self):
        @parameter()
        def fn():
            return 0.0

        self.assertTrue(hasattr(fn, 'parameter_info'))

    def test_parameter_info_default_name(self):
        @parameter()
        def fn():
            return 0.0

        self.assertEqual(fn.parameter_info.name, 'fn')

    def test_parameter_info_custom_name(self):
        @parameter(parameter_name='test_name')
        def fn():
            return 0.0

        self.assertEqual(fn.parameter_info.name, 'test_name')

    def test_parameter_info_type(self):
        @parameter(parameter_type=gym.spaces.Box(-1.0, 1.0))
        def fn():
            return 0.0

        self.assertEqual(fn.parameter_info.type, gym.spaces.Box(-1.0, 1.0))


# TODO add tests for test_* functions


if __name__ == '__main__':
    unittest.main()
