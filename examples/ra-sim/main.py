from argparse import ArgumentParser
from time import perf_counter

import gymnasium as gym

import sim
from ext import IEEE_802_11_ax_RA
from utils import ResultsManager, params_str_template, results_str_template

from reinforced_lib import RLib
from reinforced_lib.agents.mab import *


def run(
        ra_sim_args: dict[str, any],
        agent_type: type,
        agent_params: dict[str, any],
        seed: int
) -> tuple[str, float, float, float]:
    """
    Run a simulation in the ra-sim simulator.

    Parameters
    ----------
    ra_sim_args : dict
        Arguments passed to the simulator.
    agent_type : type
        Type of the selected agent.
    agent_params : dict
        Parameters of the agent.
    seed : int
        Integer used as the random key.

    Returns
    -------
    tuple[str, float, float, float]
        Tuple containing results in CSV format, mean MCS, mean data rate, and estimated mean throughput.
    """

    rl = RLib(
        agent_type=agent_type,
        agent_params=agent_params,
        ext_type=IEEE_802_11_ax_RA
    )
    rl.init(seed)

    env = gym.make('RASimEnv-v1')
    state, _ = env.reset(seed=seed, options=ra_sim_args)
    terminal = False

    results = ResultsManager(ra_sim_args, seed)

    while not terminal:
        action = rl.sample(**state)
        state, _, terminal, *_ = env.step(action)

        results.update(action, state)

    return results.summary()


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--csv_path', type=str)
    args.add_argument('--initial_position', default=1.0, type=float)
    args.add_argument('--log_every', default=1.0, type=float)
    args.add_argument('--n_wifi', default=1, type=int)
    args.add_argument('--seed', default=42, type=int)
    args.add_argument('--simulation_time', default=25.0, type=float)
    args.add_argument('--velocity', default=2.0, type=float)
    args.add_argument('--wifi_manager_name', default='RLib', type=str)

    args = args.parse_args()

    ra_sim_args = {
        'initial_position': args.initial_position,
        'log_every': args.log_every,
        'n_wifi': args.n_wifi,
        'simulation_time': args.simulation_time,
        'velocity': args.velocity,
        'wifi_manager_name': args.wifi_manager_name
    }

    print(params_str_template.format(args.n_wifi, args.simulation_time, args.initial_position, args.velocity))

    start = perf_counter()
    csv_results, mcs, rate, thr = run(ra_sim_args, ThompsonSampling, {'decay': 1.0}, args.seed)
    end = perf_counter()

    print(results_str_template.format(end - start, thr, rate, mcs))
    print(csv_results)

    if args.csv_path:
        with open(args.csv_path, 'w') as file:
            file.write(csv_results)
