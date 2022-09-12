from argparse import ArgumentParser
from time import perf_counter
from typing import Any, Dict, Tuple

import gym
gym.logger.set_level(40)

import sim
from reinforced_lib import RLib
from reinforced_lib.agents import ThompsonSampling
from reinforced_lib.exts import IEEE_802_11_ax


def run(
        ra_sim_args: Dict[str, Any],
        log_every: float,
        wifi_manager_name: str,
        seed: int
) -> Tuple[str, float, float, float]:
    """
    Run example simulation in the Rate Adaptation Simulator.

    Parameters
    ----------
    ra_sim_args : dict
        Arguments passed to the simulator.
    log_every : float
        Time interval between successive measurements.
    wifi_manager_name : str
        Name of the Wi-Fi manager in CSV.
    seed : int
         An integer used as the random key.

    Returns
    -------
    results : tuple[str, float, float, float]
        Tuple containing all results in CSV format, mean MCS, mean data rate and estimated mean throughput.
    """

    rl = RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax
    )
    rl.init(seed)

    env = gym.make('RASimEnv-v1')
    state, _ = env.reset(seed=seed, options=ra_sim_args)
    terminated = False

    csv_results = 'wifiManager,seed,nWifi,channelWidth,minGI,velocity,position,time,meanMcs,meanRate,throughput\n'
    rates = IEEE_802_11_ax().rates()

    packets_num, mcs_sum, rate_sum, thr_sum = 0, 0.0, 0.0, 0.0
    last_packets_num, last_mcs_sum, last_rate_sum, last_thr_sum = 0, 0.0, 0.0, 0.0
    last_log_time = 0.0

    while not terminated:
        action = rl.sample(**state)
        state, _, terminated, *_ = env.step(action)

        packets_num += 1
        mcs_sum += action
        rate_sum += rates[action]
        thr_sum += state['n_successful'] * rates[action]

        if state['time'] - last_log_time > log_every:
            position = state['time'] * ra_sim_args['velocity'] + ra_sim_args['initial_position']
            position, time = round(position, 2), round(state['time'], 2)

            packets = packets_num - last_packets_num
            mcs = (mcs_sum - last_mcs_sum) / packets
            rate = (rate_sum - last_rate_sum) / packets
            thr = (thr_sum - last_thr_sum) / packets

            csv_results += f'{wifi_manager_name},{seed},{ra_sim_args["n_wifi"]},20,3200,{ra_sim_args["velocity"]},' \
                           f'{position},{time},{mcs},{rate},{thr}\n'

            last_packets_num, last_mcs_sum, last_rate_sum, last_thr_sum = packets_num, mcs_sum, rate_sum, thr_sum
            last_log_time = state['time']

    return csv_results, mcs_sum / packets_num, rate_sum / packets_num, thr_sum / packets_num


params_str_template = """
Simulating an IEEE 802.11ax device with the following settings:
- frequency band: 5 GHz
- max aggregated data rate: 125 Mb/s
- channel width: 20 Mhz
- shortest guard interval: 3200 ns
- number of transmitting stations: {0}
- simulation time: {1} s
- initial AP position: {2} m
- AP velocity: {3} m/s

Starting simulation..."""

results_str_template = """Done!
Elapsed time: {0} s

Network throughput: {1} Mb/s
Mean rate: {2} Mb/s
Mean MCS: {3}
"""


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--csv_path', type=str)
    args.add_argument('--initial_position', default=1.0, type=float)
    args.add_argument('--log_every', default=1.0, type=float)
    args.add_argument('--n_wifi', default=1, type=int)
    args.add_argument('--seed', default=42, type=int)
    args.add_argument('--simulation_time', default=25.0, type=float)
    args.add_argument('--velocity', default=2.0, type=float)
    args.add_argument('--wifi_manager_name', default='RLib', type=str)

    args = args.parse_args()

    ra_sim_args = {
        'initial_position': args.initial_position,
        'n_wifi': args.n_wifi,
        'simulation_time': args.simulation_time,
        'velocity': args.velocity,
    }

    print(params_str_template.format(args.n_wifi, args.simulation_time, args.initial_position, args.velocity))

    start = perf_counter()
    csv_results, mcs, rate, thr = run(ra_sim_args, args.log_every, args.wifi_manager_name, args.seed)
    end = perf_counter()

    print(results_str_template.format(end - start, thr, rate, mcs))
    print(csv_results)

    if args.csv_path:
        with open(args.csv_path, 'w') as file:
            file.write(csv_results)
