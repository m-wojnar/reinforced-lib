import unittest
import os
import jax
import jax.numpy as jnp

from reinforced_lib.agents.mab import EGreedy
from reinforced_lib.rlib import RLib
from reinforced_lib.logs import *


class TestRLibSerialization(unittest.TestCase):

    checkpoint_path = os.path.join(os.path.expanduser("~"), "checkpoint.pkl.lz4")
    arms_probs = jnp.array([1.0, 1.0, 0.99, 0.97, 0.91, 0.77, 0.32, 0.05, 0.01, 0.0, 0.0, 0.0])
    time = jnp.linspace(0, 10, 1000)
    t_change = jnp.max(time) / 2
    key = jax.random.PRNGKey(42)


    def run_experiment(self, reload: bool, new_decay: float = None) -> list[int]:
        rl = RLib(
            agent_type=EGreedy,
            agent_params={'n_arms': len(self.arms_probs), 'e': 0.1},
            no_ext_mode=True,
            logger_types=CsvLogger,
            logger_sources=['n_failed', 'n_successful', ('action', SourceType.METRIC)],
            logger_params={'csv_path': f'output_reload={reload}_new-decay={new_decay}.csv'}
        )

        actions = []
        a = 0

        reloaded = not reload
        for t in self.time:
            r = int(jax.random.uniform(self.key) < self.arms_probs[a])
            observations = {
                'action': a,
                'reward': r
            }

            a = rl.sample(update_observations=observations)
            actions.append(a)

            if t > self.t_change and not reloaded:
                save_path = rl.save()

                if new_decay:
                    rl = RLib.load(save_path, agent_params={'n_arms': len(self.arms_probs), 'e': 0.5})
                else:
                    rl = RLib.load(save_path)
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
