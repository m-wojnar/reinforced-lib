import unittest

import jax

from reinforced_lib.agents.mab.scheduler import RoundRobinScheduler, RandomScheduler


class SchTestCase(unittest.TestCase):

    def test_rr(self):
        rr = RoundRobinScheduler(n_arms=6, initial_item=2)
        init_key, update_key, sample_key = jax.random.split(jax.random.key(4), 3)
        s = rr.init(init_key)
        s = rr.update(s, update_key)
        self.assertAlmostEqual(s.item, 3)
        a = rr.sample(s, sample_key)
        self.assertAlmostEqual(a, 3)

    def test_rand(self):
        rs = RandomScheduler(n_arms=10)
        init_key, update_key = jax.random.split(jax.random.key(4))
        s = rs.init(init_key)
        s = rs.update(s, update_key)


if __name__ == '__main__':
    unittest.main()
