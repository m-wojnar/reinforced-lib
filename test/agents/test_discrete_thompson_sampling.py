import unittest

import jax
import jax.numpy as jnp
import numpy as np

from reinforced_lib import RLib
from reinforced_lib.exts import BasicMab
from reinforced_lib.agents.mab import DiscreteThompsonSampling


OUTCOMES = jnp.array([0.0, 1.0, 2.0])
ALPHA = jnp.array([1.0, 1.0, 1.0])


class DiscreteThompsonSamplingTestCase(unittest.TestCase):
    def test_cycle(self):
        agent = DiscreteThompsonSampling(n_arms=4, alpha=ALPHA, outcomes=OUTCOMES)
        k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
        state = agent.init(k1)
        next_state = agent.update(state, k2, action=2, reward=1.0)
        a = agent.sample(next_state, k3)
        self.assertIsInstance(int(a), int)

    def test_init_shape(self):
        n_arms = 5
        agent = DiscreteThompsonSampling(n_arms=n_arms, alpha=ALPHA, outcomes=OUTCOMES)
        state = agent.init(jax.random.key(1))
        self.assertEqual(state.alpha.shape, (n_arms, len(OUTCOMES)))

    def test_update_increments_alpha(self):
        agent = DiscreteThompsonSampling(n_arms=3, alpha=ALPHA, outcomes=OUTCOMES)
        state = agent.init(jax.random.key(2))
        alpha_before = state.alpha[1].copy()
        next_state = agent.update(state, jax.random.key(3), action=1, reward=2.0)
        # outcome 2.0 is at index 2 in OUTCOMES
        self.assertEqual(next_state.alpha[1, 2], alpha_before[2] + 1.0)
        # other outcome counts for this arm should be unchanged
        np.testing.assert_array_equal(next_state.alpha[1, :2], alpha_before[:2])
        # other arms should be unchanged
        np.testing.assert_array_equal(next_state.alpha[0], state.alpha[0])

    def test_action_in_range(self):
        n_arms = 6
        agent = DiscreteThompsonSampling(n_arms=n_arms, alpha=ALPHA, outcomes=OUTCOMES)
        state = agent.init(jax.random.key(10))
        for i in range(20):
            key = jax.random.key(i + 100)
            a = agent.sample(state, key)
            self.assertGreaterEqual(int(a), 0)
            self.assertLess(int(a), n_arms)

    def test_learns_best_arm(self):
        n_arms = 3
        # arm 1 always gives reward 2.0 (highest outcome), others give 0.0
        agent = DiscreteThompsonSampling(n_arms=n_arms, alpha=ALPHA, outcomes=OUTCOMES)
        k = jax.random.key(42)
        k, init_key = jax.random.split(k)
        state = agent.init(init_key)

        rewards = [0.0, 2.0, 0.0]
        for i in range(300):
            k, sample_key, update_key = jax.random.split(k, 3)
            a = agent.sample(state, sample_key)
            state = agent.update(state, update_key, action=int(a), reward=rewards[int(a)])

        # after enough updates the best arm (index 1) should dominate
        k, final_key = jax.random.split(k)
        a = agent.sample(state, final_key)
        self.assertEqual(int(a), 1)

    def test_with_reinforced_lib(self):
        rl = RLib(
            agent_type=DiscreteThompsonSampling,
            agent_params={
                'alpha': ALPHA,
                'outcomes': OUTCOMES,
            },
            ext_type=BasicMab,
            ext_params={'n_arms': 4}
        )
        a = rl.sample(reward=1.0)
        a = rl.sample(reward=2.0)


if __name__ == '__main__':
    unittest.main()
