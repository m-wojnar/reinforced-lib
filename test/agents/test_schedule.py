import unittest

from reinforced_lib.agents.mab.scheduler.random import RandomScheduler
from reinforced_lib.agents.mab.scheduler.round_robin import RoundRobinScheduler
import jax

class SchTestCase(unittest.TestCase):

    def test_rr(self):
        rr = RoundRobinScheduler(n_arms=6,starting_arm=2)
        s = rr.init(key=jax.random.key(4))
        kar=3*(None,)
        s1 = rr.update(s,*kar)
        self.assertAlmostEqual(s1.item, 3)
        a = rr.sample(s1,*kar)
        self.assertAlmostEqual(a, 3)

    def test_rand(self):
        rs = RandomScheduler(n_arms=10)
        s = rs.init(key=jax.random.key(4))
        kar = 3 * (None,)
        s1 = rs.update(s,*kar)



if __name__ == '__main__':
    unittest.main()
