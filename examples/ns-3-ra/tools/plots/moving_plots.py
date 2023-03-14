import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from common import *


MAX_DISTANCE = 55


def plot_results(ax: plt.Axes, velocity: float) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(ALL_MANAGERS) - 1))

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobilityModel == 'Distance') & (df.velocity == velocity)]

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.wifiManager == manager], 'position')

        if manager == 'Ideal':
            ax.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            ax.plot(mean.index, mean, marker='o', markersize=1, label=manager_name, c=colors[i])
            ax.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)

    ax.set_xlim((0, MAX_DISTANCE))
    ax.set_ylim((0, 125))

    ax.set_ylabel('Station throughput [Mb/s]')
    ax.set_title(fr'$\nu$ = {velocity} m/s')

    ax.grid()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    fig, axes = plt.subplots(2, 1, sharex='col')

    for velocity, ax in zip([1, 2], axes):
        plot_results(ax, velocity)

    axes[0].tick_params('x', labelbottom=False, bottom=False)
    axes[1].set_xlabel('Distance from AP [m]')
    axes[1].legend()

    plt.savefig(f'moving-thr.pdf', bbox_inches='tight')
    plt.clf()
