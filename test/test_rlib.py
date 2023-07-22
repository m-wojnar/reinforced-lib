from reinforced_lib import RLib
from reinforced_lib.agents.mab import EGreedy


if __name__ == '__main__':
    rl = RLib(
        agent_type=EGreedy,
        agent_params={'n_arms': 4, 'e': 0.1},
        no_ext_mode=True
    )

    print(rl.observation_space)

    observations = {
        'action': 3,
        'reward': 1.0
    }

    action = rl.sample(update_observations=observations)
    print(action)

    observations = {
        'action': 3,
        'reward': 1.0
    }

    action = rl.sample(update_observations=observations)
    print(action)
