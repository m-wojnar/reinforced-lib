import matplotlib.pyplot as plt
import pandas as pd

from common import *


def plot_results(window) -> None:
    df = pd.read_csv(DATA_FILE)

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        df_i = df[df["agent"] == manager]
        df_i["throughput"] = df_i["throughput"].rolling(window=window).mean() if window > 1 else df_i["throughput"]
        mean, ci_low, ci_high = get_thr_ci(df_i, 'time')
        plt.plot(mean.index, mean, markersize=3, label=manager_name, c=COLORS[i])
        plt.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=COLORS[i], linewidth=0.0)

    plt.xlim((0, 60))
    plt.ylim((0, 50))

    plt.xlabel('Time [s]')
    plt.ylabel('Aggregate throughput [Mb/s]')

    plt.grid()
    plt.legend(loc="lower right")


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['lines.linewidth'] = 0.75

    window = 10
    plot_results(window)

    plt.savefig(f'convergence-thr_w{window}.pdf', bbox_inches='tight')
    plt.clf()
