import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import *


WINDOW = 1
DATA_FILE = "/Users/wciezobka/agh/reinforced-lib/examples/ns-3-ccod/tools/plots/all_results.csv"

INTERACTION_PERIOD = 1e-2
THR_SCALE = 5 * 150 * INTERACTION_PERIOD * 10


def plot_results() -> None:
    df = pd.read_csv(DATA_FILE)

    plt.plot([-1, 0], [-2, 0], markersize=3, label="Agent", c="white")
    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        df_i = df[df["agent"] == manager]
        df_i["throughput"] = df_i["throughput"].rolling(window=WINDOW).mean() if WINDOW > 1 else df_i["throughput"]
        df_i["throughput"] = df_i["throughput"] * THR_SCALE
        mean, ci_low, ci_high = get_thr_ci(df_i, 'time')
        plt.plot(mean.index, mean, markersize=3, label=manager_name, c=COLORS[i])
        plt.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=COLORS[i], linewidth=0.0)

    plt.xlim((0, 60))
    plt.ylim((0,  THR_SCALE * 2 / 3))

    plt.xlabel('Time [s]')
    plt.ylabel('Aggregate throughput [Mb/s]')

    plt.grid()
    plt.legend(loc="lower right")


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['lines.linewidth'] = 0.75

    plot_results()

    plt.savefig(f'convergence-thr_w{WINDOW}.pdf', bbox_inches='tight')
    plt.clf()
