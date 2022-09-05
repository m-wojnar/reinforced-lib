from typing import List

import unittest
import os
import jax
import jax.numpy as jnp
import reinforced_lib as rfl

from reinforced_lib.agents import ThompsonSampling
from reinforced_lib.exts import IEEE_802_11_ax
from reinforced_lib.rlib import RLib
from reinforced_lib.logs import *


class TestRLibSerialization(unittest.TestCase):

    checkpoint_path = os.path.join(os.path.expanduser("~"), "checkpoint.pkl.lz4")
    arms_probs = jnp.array([1.0, 1.0, 0.99, 0.97, 0.91, 0.77, 0.32, 0.05, 0.01, 0.0, 0.0, 0.0])
    time = jnp.linspace(0, 10, 1000)
    t_change = jnp.max(time) / 2
    key = jax.random.PRNGKey(42)


    def run_experiment(self, reload: bool, new_decay: float = None) -> List[int]:
        rl = rfl.RLib(
            agent_type=ThompsonSampling,
            agent_params={"decay": 0.0},
            ext_type=IEEE_802_11_ax,
            loggers_type=CsvLogger,
            loggers_sources=['n_failed', 'n_successful', ('action', SourceType.METRIC)],
            loggers_params={'csv_path': f'output_reload={reload}_new-decay={new_decay}.csv'}
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
                save_path = rl.save()

                if new_decay:
                    rl = RLib.load(save_path, agent_params={"decay": new_decay}, restore_loggers=False)
                else:
                    rl = RLib.load(save_path, restore_loggers=False)
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


if __name__ == "__main__":
    unittest.main()
