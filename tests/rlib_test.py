from reinforced_lib import RLib
from reinforced_lib.agents import ParticleFilter
from reinforced_lib.exts import IEEE_802_11_ax

if __name__ == '__main__':
    rl = RLib(
        agent_type=ParticleFilter,
        agent_params={
            'min_position': 0.0,
            'max_position': 40.0,
            'particles_num': 500,
            'n_mcs': 12
        },
        ext_type=IEEE_802_11_ax
    )

    print(rl.observation_space)

    observations = {
        'time': 0.0,
        'mcs': 0,
        'n_successful': 0,
        'n_failed': 0,
        'power': 0,
        'cw': 15
    }

    action = rl.sample(**observations)
    print(action)

    observations = {
        'time': 0.0,
        'mcs': 11,
        'n_successful': 10,
        'n_failed': 0,
        'power': 0,
        'cw': 15
    }

    action = rl.sample(**observations)
    print(action)
