import matplotlib.pyplot as plt
import pandas as pd

from common import *


MAX_N_WIFI = 16


def plot_results(ax: plt.Axes, distance: float) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobilityModel == 'Distance') & (df.position == distance) & (df.velocity == 0) & (df.nWifiReal == df.nWifi)]

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.wifiManager == manager], 'nWifiReal')

        if manager == 'Ideal':
            ax.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            ax.plot(mean.index, mean, marker='o', markersize=2, label=manager_name, c=COLORS[i])
            ax.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=COLORS[i], linewidth=0.0)

    ax.set_xticks(range(0, MAX_N_WIFI + 1, 2))
    ax.set_xlim((0, MAX_N_WIFI))
    ax.set_ylim((0, 125))

    ax.set_ylabel('Aggregate throughput [Mb/s]')
    ax.set_xlabel('Number of stations')

    ax.legend()
    ax.grid()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    fig, ax = plt.subplots()

    plot_results(ax, distance=1.)

    plt.savefig(f'equal-distance-thr.pdf', bbox_inches='tight')
    plt.clf()
