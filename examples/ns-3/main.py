import pathlib
from argparse import ArgumentParser
from ctypes import *
from typing import Any, Dict

from py_interface import *

from reinforced_lib import RLib
from reinforced_lib.agents.mab import ThompsonSampling
from reinforced_lib.exts.wifi import IEEE_802_11_ax_RA


class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('power', c_double),
        ('time', c_double),
        ('cw', c_uint32),
        ('n_failed', c_uint32),
        ('n_successful', c_uint32),
        ('n_wifi', c_uint32),
        ('station_id', c_uint32),
        ('type', c_uint8)
    ]


class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('station_id', c_uint32),
        ('mcs', c_uint8)
    ]


def run(
        ns3_args: Dict[str, Any],
        mempool_key: int,
        memblock_key: int,
        mem_size: int,
        scenario: str,
        ns3_path: str,
        seed: int
) -> None:
    """
    Run a simulation in the ns-3 simulator [1]_ with the ns3-ai library [2]_.

    Parameters
    ----------
    ns3_args : dict
        Arguments passed to the ns-3 simulator.
    mempool_key : int
        Shared memory key.
    memblock_key : int
        Shared memory ID.
    mem_size : int
        Shared memory size in bytes.
    scenario : str
        Name of the selected scenario.
    ns3_path : str
        Path to the ns-3 location.
    seed : int
        Number used to initialize the JAX and the ns-3 pseudo-random number generator.

    References
    ----------
    .. [1] The ns-3 network simulator. http://www.nsnam.org/.
    .. [2] Yin, H., Liu, P., Liu, K., Cao, L., Zhang, L., Gao, Y., & Hei, X. (2020). Ns3-Ai: Fostering Artificial
       Intelligence Algorithms for Networking Research. In Proceedings of the 2020 Workshop on Ns-3 (pp. 57–64).
       Association for Computing Machinery.
    """

    rl = RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax_RA
    )

    exp = Experiment(mempool_key, mem_size, scenario, ns3_path)
    var = Ns3AIRL(memblock_key, Env, Act)

    try:
        ns3_process = exp.run(ns3_args, show_output=True)

        while not var.isFinish():
            with var as data:
                if data is None:
                    break

                if data.env.type == 0:
                    data.act.station_id = rl.init(seed)

                elif data.env.type == 1:
                    observation = {
                        'time': data.env.time,
                        'n_successful': data.env.n_successful,
                        'n_failed': data.env.n_failed,
                        'n_wifi': data.env.n_wifi,
                        'power': data.env.power,
                        'cw': data.env.cw
                    }

                    data.act.station_id = data.env.station_id
                    data.act.mcs = rl.sample(data.env.station_id, **observation)

        ns3_process.wait()
    finally:
        del exp


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--channel_width', default=20, type=int)
    args.add_argument('--csv_path', type=str)
    args.add_argument('--data_rate', default=125, type=int)
    args.add_argument('--initial_position', default=0.0, type=float)
    args.add_argument('--log_every', default=1.0, type=float)
    args.add_argument('--mempool_key', default=1234, type=int)
    args.add_argument('--min_gi', default=3200, type=int)
    args.add_argument('--ns3_path', default=f'{pathlib.Path.home()}/ns-3-dev/', type=str)
    args.add_argument('--n_wifi', default=1, type=int)
    args.add_argument('--pcap_path', type=str)
    args.add_argument('--seed', default=42, type=int)
    args.add_argument('--simulation_time', default=20.0, type=float)
    args.add_argument('--velocity', default=0.0, type=float)
    args.add_argument('--warmup_time', default=2.0, type=float)
    args.add_argument('--wifi_manager', default='ns3::RLibWifiManager', type=str)
    args.add_argument('--wifi_manager_name', default='RLib', type=str)

    args = args.parse_args()

    ns3_args = {
        'channelWidth': args.channel_width,
        'dataRate': args.data_rate,
        'initialPosition': args.initial_position,
        'logEvery': args.log_every,
        'minGI': args.min_gi,
        'nWifi': args.n_wifi,
        'simulationTime': args.simulation_time,
        'velocity': args.velocity,
        'warmupTime': args.warmup_time,
        'wifiManager': args.wifi_manager,
        'wifiManagerName': args.wifi_manager_name,
        'RngRun': args.seed,
    }

    if args.csv_path:
        ns3_args['csvPath'] = args.csv_path

    if args.pcap_path:
        ns3_args['pcapPath'] = args.pcap_path

    memblock_key = 2333
    memory_size = 128
    scenario = 'rlib-sim'

    run(ns3_args, args.mempool_key, memblock_key, memory_size, scenario, args.ns3_path, args.seed)
