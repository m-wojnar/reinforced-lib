import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import *


MAX_DISTANCE = 55

POWER_CHANGE = {
    4: [0.335737, 2.7317, 8.99681, 15.7007, 19.3904, 19.6385,
        21.8547, 28.8143, 44.0703, 48.5148, 48.5896, 48.7193,
        51.9423, 53.2739, 57.3604, 62.5914],
    8: [0.671474, 5.4634, 17.9936, 31.4014, 38.7808, 39.277,
        43.7094, 57.6286, 88.1406]
}


def plot_results(ax: plt.Axes, delta: float, interval: float, velocity: float) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobilityModel == 'Distance') & (df.delta == delta) & (df.interval == interval) & (df.velocity == velocity)]

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.wifiManager == manager], 'time')

        if manager == 'Ideal':
            ax.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            ax.plot(mean.index, mean, marker='o', markersize=1, label=manager_name, c=COLORS[i])
            ax.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=COLORS[i], linewidth=0.0)

    for i, x in enumerate(POWER_CHANGE[interval]):
        ax.axvline(x, linestyle='--', c='r', alpha=0.4, label='Power change' if i == 0 else None)

    ax.set_xlim((0, MAX_DISTANCE))
    ax.set_ylim((0, 125))

    ax.set_ylabel('Station throughput [Mb/s]')
    ax.set_title(fr'$\Delta$ = {delta} dB, 1/$\lambda$ = {interval} s')

    ax.grid()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    fig, axes = plt.subplots(2, 1, sharex='col')

    for delta, interval, ax in zip([5, 15], [4, 8], axes.flatten()):
        plot_results(ax, delta, interval, velocity=0)

    axes[0].tick_params('x', labelbottom=False, bottom=False)
    axes[1].set_xlabel('Time [s]')
    axes[0].legend(ncol=2)

    plt.savefig(f'power-thr.pdf', bbox_inches='tight')
    plt.clf()
