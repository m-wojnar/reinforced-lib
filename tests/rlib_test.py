from reinforced_lib import RLib
from reinforced_lib.agents.thompson_sampling import ThompsonSampling
from reinforced_lib.envs.ieee_802_11_ax import IEEE_802_11_ax

if __name__ == '__main__':
    env = IEEE_802_11_ax()

    rl = RLib(
        agent_type=ThompsonSampling,
        agent_params={'context': env.context()},
        env_type=IEEE_802_11_ax
    )

    print(rl.observation_space)

    observations = {
        'time': 0.0,
        'mcs': 0,
        'n_successful': 0,
        'n_failed': 0
    }

    action = rl.sample(**observations)
    print(action)

    observations = {
        'time': 0.0,
        'mcs': 11,
        'n_successful': 10,
        'n_failed': 0
    }

    action = rl.sample(**observations)
    print(action)
