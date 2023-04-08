import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import *


def plot_results() -> None:
    df = pd.read_csv(DATA_FILE)
    df = df[df.scenario == 'basic']

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.agent == manager], 'nWifi')
        plt.plot(mean.index, mean, marker='o', markersize=3, label=manager_name, c=COLORS[i])
        plt.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=COLORS[i], linewidth=0.0)

    plt.xlim((0, 55))
    plt.ylim((0, 50))

    plt.xlabel('Number of stations')
    plt.ylabel('Aggregate throughput [Mb/s]')

    plt.grid()
    plt.legend(loc='lower left')


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['lines.linewidth'] = 0.75

    plot_results()

    plt.savefig(f'basic-thr.pdf', bbox_inches='tight')
    plt.clf()
