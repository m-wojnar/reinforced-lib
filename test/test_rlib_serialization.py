from typing import List

import unittest
import reinforced_lib as rfl
import jax
import jax.numpy as jnp

from reinforced_lib.agents import ThompsonSampling
from reinforced_lib.exts import IEEE_802_11_ax


class TestRLibSerialization(unittest.TestCase):

    checkpoint_path = "/Users/wciezobka/agh/reinforced-lib/saves/checkpoint.pkl.lz4"
    arms_probs = jnp.array([1.0, 1.0, 0.99, 0.97, 0.91, 0.77, 0.32, 0.05, 0.01, 0.0, 0.0, 0.0])
    time = jnp.linspace(0, 10, 1000)
    t_change = jnp.max(time) / 2
    key = jax.random.PRNGKey(42)


    def run_experiment(self, reload: bool, full_reload: bool = False, new_decay: int = None) -> List[int]:
        rl = rfl.RLib(
            agent_type=ThompsonSampling,
            agent_params={"decay": 0.0},
            ext_type=IEEE_802_11_ax
        )

        actions = []
        a = 0

        reloaded = not reload
        for t in self.time:
            r = int(jax.random.uniform(self.key) < self.arms_probs[a])
            observations = {
                'time': t,
                'mcs': a,
                'n_successful': r,
                'n_failed': 1 - r,
            }

            a = rl.sample(**observations)
            actions.append(int(a))

            if t > self.t_change and not reloaded:
                rl.save()

                if full_reload:
                    rl.__del__()
                    rl = rfl.RLib()

                if new_decay:
                    rl.load(self.checkpoint_path, agent_params={"decay": new_decay})
                else:
                    rl.load(self.checkpoint_path)
                reloaded = True
        
        return actions
    

    def test_reload(self):
        """
        Tests if the experiment state is fully reconstructable after reload.
        """

        actions_straight = self.run_experiment(reload=False)
        actions_reload = self.run_experiment(reload=True)
        self.assertTrue(jnp.array_equal(actions_straight, actions_reload))
    

    def test_params_alter(self):
        """
        Tests if the experiment alters after the parameter alters.
        """

        actions_straight = self.run_experiment(reload=False)
        actions_reload = self.run_experiment(reload=True, new_decay=2.0)
        self.assertFalse(jnp.array_equal(actions_straight, actions_reload))
    

    def test_full_reload(self):
        """
        Tests if the reinitialization of RLib class does not mess up everything
        """

        actions_straight = self.run_experiment(reload=False)
        actions_reload = self.run_experiment(reload=True, full_reload=True)
        self.assertTrue(jnp.array_equal(actions_straight, actions_reload))


if __name__ == "__main__":
    unittest.main()
