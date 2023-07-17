import os

import matplotlib.pylab as pl
import numpy as np
import pandas as pd
from scipy.stats import t


TOOLS_DIR = os.getenv('RLIB_DIR', os.path.join(os.path.expanduser("~"), 'reinforced-lib/examples/ns-3-ccod'))
DATA_FILE = os.path.join(TOOLS_DIR, 'outputs', 'all_results.csv')

ALL_MANAGERS = {
    'CSMA': 'Standard 802.11',
    'DQN': 'CCOD w/ DQN',
    'DDPG': 'CCOD w/ DDPG'
}
CONFIDENCE_INTERVAL = 0.99

COLUMN_WIDTH = 3.5
COLUMN_HIGHT = 2 * COLUMN_WIDTH / (1 + np.sqrt(5))
PLOT_PARAMS = {
    'figure.figsize': (COLUMN_WIDTH, COLUMN_HIGHT),
    'figure.dpi': 72,
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': 'cm',
    'axes.titlesize': 9,
    'axes.linewidth': 0.5,
    'grid.alpha': 0.42,
    'grid.linewidth': 0.5,
    'legend.title_fontsize': 6,
    'legend.fontsize': 6,
    'lines.linewidth': 0.5,
    'text.usetex': True,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}

COLORS = pl.cm.viridis(np.linspace(0., 0.75, len(ALL_MANAGERS)))


def get_thr_ci(
        data: pd.DataFrame,
        column: str,
        ci_interval: float = CONFIDENCE_INTERVAL
) -> tuple[pd.Series, pd.Series, pd.Series]:
    data = data.groupby([column])['throughput']

    measurements = data.count()
    mean = data.mean()
    std = data.std()

    alpha = 1 - ci_interval
    z = t.ppf(1 - alpha / 2, measurements - 1)

    ci_low = mean - z * std / np.sqrt(measurements)
    ci_high = mean + z * std / np.sqrt(measurements)

    return mean, ci_low, ci_high
