import importlib
import unittest


class MabExportsTestCase(unittest.TestCase):
    def test_discrete_thompson_sampling_reexport(self):
        try:
            discrete_module = importlib.import_module('reinforced_lib.agents.mab.discrete_thompson_sampling')
        except ModuleNotFoundError:
            self.skipTest('DiscreteThompsonSampling module is not available in this branch.')

        mab_module = importlib.import_module('reinforced_lib.agents.mab')

        self.assertTrue(hasattr(mab_module, 'DiscreteThompsonSampling'))
        self.assertIs(mab_module.DiscreteThompsonSampling, discrete_module.DiscreteThompsonSampling)
