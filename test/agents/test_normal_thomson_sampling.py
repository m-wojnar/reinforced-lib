import unittest

import jax
import numpy as np

from reinforced_lib import RLib
from reinforced_lib.exts import BasicMab
from reinforced_lib.agents.mab import NormalThompsonSampling, LogNormalThompsonSampling


class TSTestCase(unittest.TestCase):
    def test_normal_cycle(self):
        ts = NormalThompsonSampling(n_arms=8, alpha=1., beta=1., lam=2., mu=0.)
        k1, k2, k3 = jax.random.split(jax.random.key(4), 3)
        state = ts.init(k1)
        next_state = ts.update(state, k2, action=3, reward=3.0)
        a = ts.sample(next_state, k3)

    def test_lognormal_cycle(self):
        ts = LogNormalThompsonSampling(n_arms=8, alpha=1., beta=1., lam=2., mu=0.)
        k1, k2, k3 = jax.random.split(jax.random.key(4), 3)
        state = ts.init(k1)
        next_state = ts.update(state, k2, action=3, reward=3.0)
        a = ts.sample(next_state, k3)

    def test_agent(self):
        k = jax.random.PRNGKey(42)
        k, sample_key = jax.random.split(k)

        ts = NormalThompsonSampling(n_arms=2, alpha=1., beta=1., lam=2., mu=0.)
        k, init_key = jax.random.split(k)
        state = ts.init(init_key)

        for i in range(500):
            k, ts_sample_key, ts_update_key, env_key = jax.random.split(k, 4)
            a = ts.sample(state, ts_sample_key)
            env = np.asarray([5, 20.]) + jax.random.normal(env_key, shape=(2,)) * 3
            r = env[a]
            state = ts.update(state, ts_update_key, action=a, reward=r)

        self.assertTrue(state.mu[1] > state.mu[0])

    def test_normal_with_reinforced_lib(self):
        rl = RLib(
            agent_type=NormalThompsonSampling,
            agent_params={'alpha': 1., 'beta': 1., 'lam': 2., 'mu': 0.},
            ext_type=BasicMab,
            ext_params={'n_arms': 4}
        )
        a = rl.sample(reward=1.0)
        a = rl.sample(reward=1.0)

    def test_lognormal_with_reinforced_lib(self):
        rl = RLib(
            agent_type=LogNormalThompsonSampling,
            agent_params={'alpha': 1., 'beta': 1., 'lam': 2., 'mu': 0.},
            ext_type=BasicMab,
            ext_params={'n_arms': 4}
        )
        a = rl.sample(reward=1.0)
        a = rl.sample(reward=1.0)


if __name__ == '__main__':
    unittest.main()
