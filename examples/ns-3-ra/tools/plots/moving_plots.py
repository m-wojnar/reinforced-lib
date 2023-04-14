import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import *


MAX_DISTANCE = 55


def plot_results(ax: plt.Axes, velocity: float) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobilityModel == 'Distance') & (df.velocity == velocity)]

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.wifiManager == manager], 'position')

        if manager == 'Ideal':
            ax.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            ax.plot(mean.index, mean, marker='o', markersize=1, label=manager_name, c=COLORS[i])
            ax.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=COLORS[i], linewidth=0.0)

    ax.set_xlim((0, MAX_DISTANCE))
    ax.set_ylim((0, 125))

    ax.set_ylabel('Station throughput [Mb/s]')
    ax.set_xlabel('Distance from AP [m]')

    ax.legend()
    ax.grid()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    fig, ax = plt.subplots()

    plot_results(ax, velocity=1.)

    plt.savefig(f'moving-thr.pdf', bbox_inches='tight')
    plt.clf()
