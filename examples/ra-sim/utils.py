from typing import Any, Dict, Tuple

from reinforced_lib.exts.wifi import IEEE_802_11_ax_RA


class ResultsManager:
    """
    Collects the agent's statistics and the performance results while running a simulation.

    Parameters
    ----------
    ra_sim_args : dict
        Arguments passed to the simulator.
    seed : int
        Integer used as the random key.
    """

    def __init__(self, ra_sim_args: Dict[str, Any], seed: int) -> None:
        self.wifi_ext = IEEE_802_11_ax_RA()

        self.csv_results = 'wifiManager,seed,nWifi,channelWidth,minGI,velocity,position,time,meanMcs,meanRate,throughput\n'
        self.log_str_template = f'{ra_sim_args["wifi_manager_name"]},{seed},{ra_sim_args["n_wifi"]},20,3200,{ra_sim_args["velocity"]},' + '{0},{1},{2},{3},{4}\n'

        self.packets_num = 0
        self.mcs_sum = 0.0
        self.rate_sum = 0.0
        self.thr_sum = 0.0

        self.last_packets_num = 0
        self.last_mcs_sum = 0.0
        self.last_rate_sum = 0.0
        self.last_thr_sum = 0.0

        self.initial_position = ra_sim_args['initial_position']
        self.velocity = ra_sim_args['velocity']
        self.log_every = ra_sim_args['log_every']
        self.last_log_time = 0.0

    def update(self, action: int, state: Dict[str, Any]) -> None:
        """
        Updates the simulation statistics.

        Parameters
        ----------
        action : int
            Last action taken by the agent.
        state : dict
            Current simulation state.
        """

        self.packets_num += 1
        self.mcs_sum += action
        self.rate_sum += self.wifi_ext.rates()[action]
        self.thr_sum += self.wifi_ext.reward(action, state['n_successful'], state['n_failed'])

        if state['time'] - self.last_log_time > self.log_every:
            position = state['time'] * self.velocity + self.initial_position
            position, time = round(position, 2), round(state['time'], 2)

            packets = self.packets_num - self.last_packets_num
            mcs = (self.mcs_sum - self.last_mcs_sum) / packets
            rate = (self.rate_sum - self.last_rate_sum) / packets
            thr = (self.thr_sum - self.last_thr_sum) / packets

            self.csv_results += self.log_str_template.format(position, time, mcs, rate, thr)

            self.last_packets_num = self.packets_num
            self.last_mcs_sum = self.mcs_sum
            self.last_rate_sum = self.rate_sum
            self.last_thr_sum = self.thr_sum
            self.last_log_time = state['time']

    def summary(self) -> Tuple[str, float, float, float]:
        """
        Calculates the simulation summaries and returns results.

        Returns
        -------
        tuple[str, float, float, float]
            Tuple containing results in CSV format, mean MCS, mean data rate, and estimated mean throughput.
        """

        mean_mcs = self.mcs_sum / self.packets_num
        mean_rate = self.rate_sum / self.packets_num
        mean_thr = self.thr_sum / self.packets_num

        return self.csv_results, mean_mcs, mean_rate, mean_thr


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
