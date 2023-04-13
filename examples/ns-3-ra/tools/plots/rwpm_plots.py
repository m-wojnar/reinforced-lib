import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import *


N_WIFI = 10


def plot_results(ax: plt.Axes, velocity: float) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobilityModel == 'RWPM') & (df.nWifiReal == N_WIFI) & (df.velocity == velocity)]

    sns.violinplot(
        ax=ax, data=df, x='wifiManager', y='throughput',
        order=ALL_MANAGERS.keys(), palette=COLORS.tolist()
    )

    ax.set_ylim((0, 90))
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])

    ax.set_ylabel('Aggregate throughput [Mb/s]')
    ax.set_xlabel('')

    ax.set_axisbelow(True)
    ax.grid(axis='y')


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['figure.figsize'] = (COLUMN_WIDTH, 1.4 * COLUMN_WIDTH)

    fig, axes = plt.subplots(2, 1, sharex='col')

    for velocity, ax in zip([0., 1.4], axes):
        plot_results(ax, velocity)

    axes[0].tick_params('x', labelbottom=False, bottom=False)
    axes[1].set_xticklabels(ALL_MANAGERS.values())
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    axes[0].set_title('Static stations')
    axes[1].set_title('Mobile stations')

    plt.savefig(f'rwpm-thr.pdf', bbox_inches='tight')
    plt.clf()
