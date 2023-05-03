from argparse import ArgumentParser
from ctypes import *
from typing import Any, Dict

from py_interface import *

from reinforced_lib import RLib
from reinforced_lib.agents.mab import *
from reinforced_lib.agents.wifi import *
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


memblock_key = 2333
memory_size = 128
simulation = 'ra-sim'


def run(
        ns3_args: Dict[str, Any],
        ns3_path: str,
        mempool_key: int,
        agent_type: type,
        agent_params: Dict[str, Any],
        seed: int
) -> None:
    """
    Run a simulation in the ns-3 simulator [1]_ with the ns3-ai library [2]_.

    Parameters
    ----------
    ns3_args : dict
        Arguments passed to the ns-3 simulator.
    ns3_path : str
        Path to the ns-3 location.
    mempool_key : int
        Shared memory key.
    agent_type : type
        Type of the selected agent.
    agent_params : dict
        Parameters of the agent.
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
        agent_type=agent_type,
        agent_params=agent_params,
        ext_type=IEEE_802_11_ax_RA
    )

    exp = Experiment(mempool_key, memory_size, simulation, ns3_path)
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
                    data.act.mcs = rl.sample(agent_id=data.env.station_id, **observation)

        ns3_process.wait()
    finally:
        del exp


if __name__ == '__main__':
    args = ArgumentParser()

    args.add_argument('--agent', required=True, type=str)
    args.add_argument('--area', default=40.0, type=float)
    args.add_argument('--channelWidth', default=20, type=int)
    args.add_argument('--csvPath', type=str)
    args.add_argument('--deltaPower', default=0.0, type=float)
    args.add_argument('--dataRate', default=125, type=int)
    args.add_argument('--initialPosition', default=0.0, type=float)
    args.add_argument('--intervalPower', default=4.0, type=float)
    args.add_argument('--logEvery', default=1.0, type=float)
    args.add_argument('--lossModel', default='LogDistance', type=str)
    args.add_argument('--mempoolKey', default=1234, type=int)
    args.add_argument('--minGI', default=3200, type=int)
    args.add_argument('--mobilityModel', required=True, type=str)
    args.add_argument('--nodeSpeed', default=1.4, type=float)
    args.add_argument('--nodePause', default=20.0, type=float)
    args.add_argument('--ns3Path', required=True, type=str)
    args.add_argument('--nWifi', default=1, type=int)
    args.add_argument('--pcapPath', type=str)
    args.add_argument('--seed', default=42, type=int)
    args.add_argument('--simulationTime', default=20.0, type=float)
    args.add_argument('--velocity', default=0.0, type=float)
    args.add_argument('--warmupTime', default=2.0, type=float)
    args.add_argument('--wifiManagerName', default='RLib', type=str)

    args = vars(args.parse_args())

    args['RngRun'] = args['seed']
    agent = args.pop('agent')

    agent_type = {
        'EGreedy': EGreedy,
        'Exp3': Exp3,
        'Softmax': Softmax,
        'ThompsonSampling': ThompsonSampling,
        'UCB': UCB,
        'ParticleFilter': ParticleFilter
    }
    default_params = {
        'EGreedy': {'e': 0.008, 'alpha': 0.032, 'optimistic_start': 64.0},
        'Exp3': {'gamma': 0.128},
        'Softmax': {'lr': 2.048, 'alpha': 0.016, 'tau': 4.0, 'multiplier': 0.01},
        'ThompsonSampling': {'decay': 2.0},
        'UCB': {'c': 16.0, 'gamma': 0.996},
        'ParticleFilter': {'scale': 4.0, 'num_particles': 900}
    }

    run(args, args.pop('ns3Path'), args.pop('mempoolKey'), agent_type[agent], default_params[agent], args.pop('seed'))
