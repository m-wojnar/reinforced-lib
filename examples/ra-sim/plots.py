import warnings
warnings.filterwarnings('ignore')

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


COLUMN_WIDTH = 7.0
COLUMN_HEIGHT = 2 * COLUMN_WIDTH / (1 + jnp.sqrt(5))
PLOT_PARAMS = {
    'figure.figsize': (COLUMN_WIDTH, COLUMN_HEIGHT),
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


def ra_sim_eval():
    gammas = pd.read_csv('ucb_params.csv')['gamma'].unique()
    gammas.sort()

    df_ns3 = pd.read_csv('ucb_ns3_params.csv').groupby(['c', 'gamma'], as_index=False)['thr'].mean().pivot('c', 'gamma', 'thr')
    df_ra = pd.read_csv('ucb_params.csv').groupby(['c', 'gamma'], as_index=False)['thr'].mean().pivot('c', 'gamma', 'thr')

    fig, axs = plt.subplots(1, 2, sharey='row', figsize=(COLUMN_WIDTH / 2, COLUMN_HEIGHT / 2))
    cbar_ax = fig.add_axes([0.9, 0.313, 0.015, 0.475])

    sns.heatmap(df_ns3, ax=axs[0], vmin=20, vmax=45, cbar_ax=cbar_ax, cmap='viridis', square=True)
    axs[0].set_title('ns-3')
    axs[0].set_xlabel(r'Discount factor $\gamma$')
    axs[0].set_ylabel(r'Degree of exploration c')
    axs[0].set_xticks(jnp.arange(len(gammas)) + 0.5)
    axs[0].set_xticklabels(gammas)
    axs[0].tick_params(axis='both', which='major', labelsize=7)

    sns.heatmap(df_ra, ax=axs[1], vmin=20, vmax=45, cbar=False, cmap='viridis', square=True)
    axs[1].set_title('ra-sim')
    axs[1].set_xlabel(r'Discount factor $\gamma$')
    axs[1].set_xticks(jnp.arange(len(gammas)) + 0.5)
    axs[1].set_xticklabels(gammas, fontsize=7)
    axs[1].yaxis.set_visible(False)

    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig('ra-sim_eval.pdf', bbox_inches='tight')
    plt.show()


def print_best_params(name, df):
    cols = df.columns[:-2].to_list()
    values = df.groupby(cols)['thr'].mean().sort_values().index[-1]

    print(f'Best {name} params:')

    if len(cols) == 1:
        values = [values]

    for c, v in zip(cols, values):
        print(f'{c} = {v}')

    print()


def e_greedy():
    df = pd.read_csv('e-greedy_params.csv')
    print_best_params('e-greedy', df)

    fig, axs = plt.subplots(2, 3, sharey='row', sharex='col')
    cbar_ax = fig.add_axes([0.92, 0.16, 0.012, 0.77])

    for i, (os, ax) in enumerate(zip(df['optimistic_start'].unique(), axs.flatten())):
        data = df[df.optimistic_start == os]
        data = data.groupby(['e', 'alpha'], as_index=False)['thr'].mean().pivot('e', 'alpha', 'thr')

        sns.heatmap(data, ax=ax, vmin=4, vmax=48, cbar_ax=cbar_ax, cmap='viridis', square=True, xticklabels=True, yticklabels=True)
        ax.set_title(fr'$\nu$ = {os}')

        if i % 3 == 0:
            ax.set_ylabel(r'Experiment rate $\epsilon$')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.yaxis.set_visible(False)

        if i >= 3:
            ax.set_xlabel(r'Step size $\alpha$')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        else:
            ax.xaxis.set_visible(False)

    fig.tight_layout(rect=[0, 0, .92, 1])
    plt.savefig('e-greedy_params.pdf', bbox_inches='tight')
    plt.show()


def exp3():
    df = pd.read_csv('exp3_params.csv')
    print_best_params('exp3', df)

    plt.figure(figsize=(0.19 * COLUMN_WIDTH, 0.3 * COLUMN_WIDTH))

    df = df.groupby('gamma')['thr'].mean().to_frame()
    sns.heatmap(df, xticklabels=False, cmap='viridis', square=True)
    plt.ylabel(r'Exploration factor $\gamma$')
    plt.savefig('exp3_params.pdf', bbox_inches='tight')
    plt.show()


def ts():
    df = pd.read_csv('ts_params.csv')
    print_best_params('TS', df)

    plt.figure(figsize=(0.19 * COLUMN_WIDTH, 0.3 * COLUMN_WIDTH))

    df = df.groupby('decay')['thr'].mean().to_frame()
    sns.heatmap(df, xticklabels=False, cmap='viridis', square=True)
    plt.ylabel(r'Decay $w$')
    plt.savefig('ts_params.pdf', bbox_inches='tight')
    plt.show()


def ucb():
    df = pd.read_csv('ucb_params.csv')
    print_best_params('UCB', df)

    plt.figure(figsize=(COLUMN_WIDTH / 2, COLUMN_HEIGHT / 2))

    df = df.groupby(['c', 'gamma'], as_index=False)['thr'].mean().pivot('c', 'gamma', 'thr')
    ax = sns.heatmap(df, cmap='viridis', square=True)
    plt.xlabel(r'Discount factor $\gamma$')
    plt.ylabel(r'Degree of exploration c')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig('ucb_params.pdf', bbox_inches='tight')
    plt.show()


def softmax():
    df = pd.read_csv('softmax_params.csv')
    print_best_params('Softmax', df)

    fig, axs = plt.subplots(5, 4, sharey='row', sharex='col', figsize=(COLUMN_WIDTH, COLUMN_WIDTH))
    cbar_ax = fig.add_axes([0.92, 0.102, 0.012, 0.85])

    for i, lr in enumerate(df['lr'].unique()):
        for j, mul in enumerate(df['multiplier'].unique()):
            data = df[(df.lr == lr) & (df.multiplier == mul)]
            data = data.groupby(['tau', 'alpha'], as_index=False)['thr'].mean().pivot('tau', 'alpha', 'thr')

            ax = axs[i, j]
            sns.heatmap(data, ax=ax, vmin=20, vmax=45, cbar_ax=cbar_ax, cmap='viridis', square=True, xticklabels=True, yticklabels=True)
            ax.set_title(fr'$lr$ = {lr}, $mul$ = {mul}')

            if j == 0:
                ax.set_ylabel(r'Temperature $\tau$')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            else:
                ax.yaxis.set_visible(False)

            if i == 4:
                ax.set_xlabel(r'Step size $\alpha$')
            else:
                ax.xaxis.set_visible(False)

    fig.tight_layout(rect=[0, 0, .92, 1])
    plt.savefig(f'softmax_params.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    ra_sim_eval()
    e_greedy()
    exp3()
    softmax()
    ts()
    ucb()
