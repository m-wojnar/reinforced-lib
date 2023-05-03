from itertools import product

import numpy as np
from joblib import Parallel, delayed

from main import run
from reinforced_lib.agents.mab import *


def optimize_params():
    ra_sim_args = {
        'initial_position': 0.,
        'log_every': 50.,
        'n_wifi': 10,
        'simulation_time': 50.,
        'velocity': 1.,
        'wifi_manager_name': 'RLib'
    }

    seeds = list(range(100, 105))

    params_grid = [
        (EGreedy, 'e-greedy', {
            'e': np.power(2., np.arange(0, 10)) / 1000,
            'alpha': np.power(2., np.arange(0, 10)) / 1000,
            'optimistic_start': np.concatenate([[0.], np.power(2., np.arange(2, 11, 2))])
        }),
        (Exp3, 'exp3', {
            'gamma': np.power(2., np.arange(0, 10)) / 1000,
        }),
        (Softmax, 'softmax', {
            'lr':  np.power(2., np.arange(3, 12, 2)) / 1000,
            'alpha': np.power(2., np.arange(0, 10)) / 1000,
            'tau': np.power(2., np.arange(-3, 4)),
            'multiplier': np.power(10., np.arange(-3, 1))
        }),
        (ThompsonSampling, 'ts', {
            'decay': np.concatenate([[0.], np.power(2., np.arange(-4, 6))])
        }),
        (UCB, 'ucb', {
            'c': np.power(2., np.arange(-2, 7)),
            'gamma': np.power(2., np.arange(-10, 0)),
        }),
    ]

    for agent_type, agent_str, params in params_grid:
        runs = [dict(zip(params.keys(), p)) for p in product(*params.values())]
        results = Parallel(n_jobs=-1)(
            delayed(run)(ra_sim_args, agent_type, agent_params, seed) for agent_params, seed in product(runs, seeds)
        )

        file = open(f'{agent_str}_params.csv', 'w')
        file.write(','.join(params.keys()) + ',seed,thr\n')

        for (_, _, _, thr), (agent_params, seed) in zip(results, product(runs, seeds)):
            file.write(','.join(map(str, agent_params.values())) + f',{seed},{thr}\n')

        file.close()


if __name__ == '__main__':
    optimize_params()
