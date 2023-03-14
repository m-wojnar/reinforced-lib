import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from common import *


MAX_N_WIFI = 30


def plot_results(ax: plt.Axes, distance: float) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(ALL_MANAGERS) - 1))

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobilityModel == 'Distance') & (df.position == distance)]

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.wifiManager == manager], 'nWifiReal')

        if manager == 'Ideal':
            ax.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            ax.plot(mean.index, mean, marker='o', markersize=2, label=manager_name, c=colors[i])
            ax.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)

    ax.set_xlim((0, MAX_N_WIFI))
    ax.set_ylim((0, 125))

    ax.set_ylabel('Aggregate throughput [Mb/s]')
    ax.set_title(fr'$\rho$ = {distance} m')

    ax.grid()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    fig, axes = plt.subplots(1, 1, sharex='col', figsize=(COLUMN_WIDTH, COLUMN_HIGHT / 2))

    for distance, ax in zip([0], [axes]):
        plot_results(ax, distance)

    axes.set_xlabel('Number of stations')
    axes.legend()

    plt.savefig(f'equal-distance-thr.pdf', bbox_inches='tight')
    plt.clf()
