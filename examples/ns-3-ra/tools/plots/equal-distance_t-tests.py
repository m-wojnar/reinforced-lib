import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import *


def plot_results(distance: float, n_wifi: int) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df.loc[(df.mobilityModel == 'Distance') & (df.nWifiReal == n_wifi) & (df.position == distance)]

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

    plt.savefig(f'equal-distance-d{distance}-n{n_wifi}-ttest.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['figure.figsize'] = (4, 5)

    n_wifi_to_compare = 4

    for distance in [1, 20]:
        plot_results(distance, n_wifi_to_compare)
