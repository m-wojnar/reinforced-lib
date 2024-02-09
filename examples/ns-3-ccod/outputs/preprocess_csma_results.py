import os
from argparse import ArgumentParser
from glob import glob

import pandas as pd


INTERACTION_PERIOD = 1e-2
THR_SCALE = 5 * 150 * INTERACTION_PERIOD * 10


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-i', '--input', type=str, required=True)
    args.add_argument('-o', '--output', type=str, required=True)
    args = args.parse_args()

    for file in glob(os.path.join(args.input, 'CSMA_*.csv')):
        df = pd.read_csv(file, header=None)

        file = os.path.basename(file)
        _, scenario, n_wifi, _ = file.split('_')

        new_df = pd.DataFrame(columns=['agent', 'scenario', 'nWifi', 'seed', 'time', 'throughput'])
        new_df['time'] = df[1]
        new_df['throughput'] = df[2] * THR_SCALE
        new_df['agent'] = 'CSMA'
        new_df['scenario'] = scenario
        new_df['nWifi'] = n_wifi
        new_df['seed'] = 300
        new_df.to_csv(os.path.join(args.output, file), index=False, header=False)
