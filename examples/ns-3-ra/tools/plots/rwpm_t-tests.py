import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import *


N_WIFI = 10


def plot_results(velocity: float) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobilityModel == 'RWPM') & (df.nWifiReal == N_WIFI) & (df.velocity == velocity)]

    results = get_thr_ttest(df)
    mask = np.tril(np.ones_like(results))
    managers = ALL_MANAGERS.values()

    ax = sns.heatmap(
        results,
        xticklabels=managers,
        yticklabels=managers,
        annot=True,
        fmt='.3f',
        mask=mask,
        cmap='viridis',
        annot_kws={'fontsize': 5}
    )

    ax.figure.subplots_adjust(left=0.3, bottom=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.savefig(f'rwpm-v{velocity}-ttest.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    for velocity in [0., 1.4]:
        plot_results(velocity)
