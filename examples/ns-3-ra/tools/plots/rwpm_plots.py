import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pandas as pd
import seaborn as sns

from common import *


N_WIFI = 10


def plot_results(ax: plt.Axes, velocity: float) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(ALL_MANAGERS) - 1)).tolist() + ['gray']

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobilityModel == 'RWPM') & (df.nWifiReal == N_WIFI) & (df.velocity == velocity)]

    sns.violinplot(ax=ax, data=df, x='wifiManager', y='throughput', order=ALL_MANAGERS.keys(), palette=colors)
    oracle = df[df.wifiManager == 'Ideal']['throughput'].mean()
    ax.axhline(oracle, linestyle='--', c='gray', label=f'{ALL_MANAGERS["Ideal"]} mean')

    ax.set_ylim((0, 90))
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])

    ax.set_ylabel('Aggregate throughput [Mb/s]')
    ax.set_xlabel('')

    ax.set_axisbelow(True)
    ax.grid(axis='y')


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    fig, axes = plt.subplots(2, 1, sharex='col')

    for velocity, ax in zip([0., 1.4], axes):
        plot_results(ax, velocity)

    axes[0].tick_params('x', labelbottom=False, bottom=False)
    axes[1].set_xticklabels(ALL_MANAGERS.values())
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    axes[1].legend()

    axes[0].set_title('Static stations')
    axes[1].set_title('Mobile stations')

    plt.savefig(f'rwpm-thr.pdf', bbox_inches='tight')
    plt.clf()
