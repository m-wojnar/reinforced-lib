import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import *


def plot_results(velocity: float, distance: float) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df.loc[(df.mobilityModel == 'Distance') & (df.velocity == velocity) & (df.position == distance)]

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

    plt.savefig(f'moving-v{velocity}-d{distance}-ttest.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    distance_to_compare = 10

    for velocity in [1, 2]:
        plot_results(velocity, distance_to_compare)
