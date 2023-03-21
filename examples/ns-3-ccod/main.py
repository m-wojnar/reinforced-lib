# TODO adapt to CCOD scenario

from argparse import ArgumentParser
from ctypes import *
from typing import Any, Dict

from py_interface import *

from reinforced_lib import RLib
from reinforced_lib.agents.deep import *
from reinforced_lib.agents.mab import *
from reinforced_lib.agents.wifi import *
from reinforced_lib.exts.wifi import IEEE_802_11_ax_RA

MAX_HISTORY_LENGTH = 128


class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('history_sie', c_uint32),
        ('history', c_float * MAX_HISTORY_LENGTH),
        ('reward', c_float),
        ('sim_time', c_double)
    ]


class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('action', c_uint32)
    ]


memblock_key = 2333
memory_size = 2048
simulation = 'ccod-sim'


def run(
        ns3_args: Dict[str, Any],
        ns3_path: str,
        mempool_key: int,
        agent_type: type,
        agent_params: Dict[str, Any],
        seed: int,
        verbose: bool
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

    def print_environment(observation):
        print(f"Sim Time: {'{:.3f}'.format(observation['sim_time'])}\t", end="")
        print(f"Reward: {'{:.3f}'.format(observation['reward'])}\t", end="")
        print(f"History[{len(observation['history'])}]:", end="")
        for p_col in observation['history']:
            print(" {:.3f}".format(p_col), end="")
        print()

    exp = Experiment(mempool_key, memory_size, simulation, ns3_path, debug=False)
    var = Ns3AIRL(memblock_key, Env, Act)

    try:
        ns3_process = exp.run(ns3_args, show_output=True)

        while not var.isFinish():
            with var as data:
                if data is None:
                    break

                observation = {
                    'history': data.env.history[:data.env.history_sie],
                    'reward': data.env.reward,
                    'sim_time': data.env.sim_time
                }
                print_environment(observation) if verbose else None

                data.act.action = 4

        ns3_process.wait()
    finally:
        del exp


if __name__ == '__main__':
    args = ArgumentParser()

    # Python arguments
    args.add_argument('--agent', required=True, type=str)
    args.add_argument('--mempoolKey', default=1234, type=int)
    args.add_argument('--ns3Path', required=True, type=str)
    args.add_argument('--pythonSeed', default=42, type=int)

    # ns3 arguments
    args.add_argument('--agentType', default='discrete', type=str)
    args.add_argument('--CW', default=0, type=int)
    args.add_argument('--dryRun', default=False, action='store_true')
    args.add_argument('--envStepTime', default=0.1, type=float)
    args.add_argument('--historyLength', default=20, type=int)
    args.add_argument('--nonZeroStart', default=False, action='store_true')
    args.add_argument('--nWifi', default=5, type=int)
    args.add_argument('--rng', default=42, type=int)
    args.add_argument('--scenario', default='basic', type=str)
    args.add_argument('--seed', default=-1, type=int)
    args.add_argument('--simTime', default=10.0, type=float)
    args.add_argument('--tracing', default=False, action='store_true')
    args.add_argument('--verbose', default=False, action='store_true')

    args = vars(args.parse_args())

    args['RngRun'] = args['pythonSeed']
    agent = args.pop('agent')

    agent_type = {
        'EGreedy': EGreedy,
        'Exp3': Exp3,
        'Softmax': Softmax,
        'ThompsonSampling': ThompsonSampling,
        'UCB': UCB,
        'ParticleFilter': ParticleFilter,
    }
    default_params = {
        'EGreedy': {'e': 0.001, 'alpha': 0.5, 'optimistic_start': 32.0},
        'Exp3': {'gamma': 0.15},
        'Softmax': {'lr': 0.256, 'alpha': 0.5, 'tau': 1.0, 'multiplier': 0.01},
        'ThompsonSampling': {'decay': 2.0},
        'UCB': {'c': 16.0, 'gamma': 0.996},
        'ParticleFilter': {'scale': 4.0, 'num_particles': 900}
    }

    run(args, args.pop('ns3Path'), args.pop('mempoolKey'), agent_type[agent], default_params[agent], args.pop('pythonSeed'), args['verbose'])
