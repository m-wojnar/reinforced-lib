import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import *


def plot_results(delta: float, interval: float, time: float) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df.loc[(df.mobilityModel == 'Distance') & (df.delta == delta) & (df.interval == interval) & (df.time == time)]

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

    plt.savefig(f'power-d{delta}-i{interval}-t{time}-ttest.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    for delta, interval, time in zip([5, 15], [4, 8], [23, 33]):
        plot_results(delta, interval, time)
