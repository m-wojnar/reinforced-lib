import unittest

import jax
import jax.numpy as jnp

from reinforced_lib.agents.mab import ThompsonSampling
from reinforced_lib.agents.mab.scheduler import RandomScheduler
from reinforced_lib.experimental.masked import Masked, MaskedState


class MaskedTestCase(unittest.TestCase):
    def test_masked(self):
        # environment characteristics
        arms_probs = jnp.array([0.1, 0.2, 0.3, 0.8, 0.3])
        context = jnp.array([5.0, 5.0, 2.0, 2.0, 5.0])

        # agent setup
        mask = jnp.asarray([0, 0, 0, 0, 1], dtype=jnp.bool)
        ts_agent = ThompsonSampling(len(arms_probs))
        agent = Masked(ts_agent, mask)

        key = jax.random.key(4)
        init_key, key = jax.random.split(key)

        state = agent.init(init_key)

        # helper variables
        delta_t = 0.01
        actions = []
        a = 0

        for _ in range(100):
            # pull selected arm
            key, random_key, update_key, sample_key = jax.random.split(key, 4)
            r = jax.random.uniform(random_key) < arms_probs[a]

            # update state and sample
            state = agent.update(state, update_key, a, r, 1 - r, delta_t)
            a = agent.sample(state, sample_key, context)

            # save selected action
            actions.append(a.item())

        self.assertNotIn(4, actions)

    def test_masked_schedule(self):
        # agent setup
        mask = jnp.asarray([0, 0, 0, 0, 1], dtype=jnp.bool)
        agent = RandomScheduler(len(mask))
        agent = Masked(agent, mask)

        key = jax.random.key(4)
        init_key, key = jax.random.split(key)

        state = agent.init(init_key)

        # helper variables
        actions = []

        for _ in range(100):
            # pull selected arm
            key, update_key, sample_key = jax.random.split(key, 3)

            # update state and sample
            state = agent.update(state, update_key)
            a = agent.sample(state, sample_key)

            # save selected action
            actions.append(a.item())

        self.assertNotIn(4, actions)

    def test_change_mask(self):
        # environment characteristics
        arms_probs = jnp.array([0.1, 0.2, 0.3, 0.8, 0.3])
        context = jnp.array([5.0, 5.0, 2.0, 2.0, 5.0])

        # agent setup
        mask = jnp.asarray([0, 0, 0, 0, 1], dtype=jnp.bool)
        ts_agent = ThompsonSampling(len(arms_probs), decay=0.01)
        agent = Masked(ts_agent, mask)

        key = jax.random.key(4)
        init_key, key = jax.random.split(key)

        state = agent.init(init_key)

        # helper variables
        delta_t = 0.01
        actions = []
        a = 0

        for _ in range(100):
            # pull selected arm
            key, random_key, update_key, sample_key = jax.random.split(key, 4)
            r = jax.random.uniform(random_key) < arms_probs[a]

            # update state and sample
            state = agent.update(state, update_key, a, r, 1 - r, delta_t)
            a = agent.sample(state, sample_key, context)

            # save selected action
            actions.append(a.item())

        self.assertIn(0, actions)
        self.assertNotIn(4, actions)

        # second agent setup
        mask = jnp.asarray([1, 0, 0, 0, 0], dtype=jnp.bool)
        agent = Masked(ts_agent, mask)
        state = MaskedState(agent_state=state.agent_state, mask=mask)

        actions = []

        for _ in range(100):
            # pull selected arm
            key, random_key, update_key, sample_key = jax.random.split(key, 4)
            r = jax.random.uniform(random_key) < arms_probs[a]

            # update state and sample
            state = agent.update(state, update_key, a, r, 1 - r, delta_t)
            a = agent.sample(state, sample_key, context)

            # save selected action
            actions.append(a.item())

        self.assertIn(4, actions)
        self.assertNotIn(0, actions)
