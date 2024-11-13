import os
from argparse import ArgumentParser
from glob import glob
from itertools import product

import pandas as pd


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-i', '--input', type=str, required=True)
    args.add_argument('-o', '--output', type=str, required=True)
    args = args.parse_args()

    for agent, n_wifi in product(['CSMA', 'DDPG', 'DDQN'], [5, 15, 30, 50]):
        new_df = pd.DataFrame(columns=['agent', 'scenario', 'nWifi', 'seed', 'time', 'throughput'])

        for file in glob(os.path.join(args.input, f'{agent}_basic_{n_wifi}_*.csv')):
            df = pd.read_csv(file, header=None)
            seed = df.iloc[0, 3]
            thr = df[5].mean()

            new_df = pd.concat([
                new_df,
                pd.DataFrame({
                    'agent': [agent],
                    'scenario': ['basic'],
                    'nWifi': [n_wifi],
                    'seed': [seed],
                    'time': [''],
                    'throughput': [thr]
                })
            ])

        new_df.to_csv(os.path.join(args.output, f'{agent}_basic_{n_wifi}.csv'), index=False, header=False)
