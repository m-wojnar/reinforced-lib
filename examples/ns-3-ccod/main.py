from argparse import ArgumentParser
from ctypes import *
from typing import Any, Dict
from chex import Array

from py_interface import *
import haiku as hk
import optax
import jax.numpy as jnp

from reinforced_lib import RLib
from reinforced_lib.agents.deep import *
from reinforced_lib.exts.wifi import IEEE_802_11_ax_CCOD

MAX_HISTORY_LENGTH = 128
NO_ACTIONS = 6


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


@hk.transform_with_state
def q_network(x: Array) -> Array:

    # sequence_length - it is the history depth to analyze, equals the number of LSTM cells
    # batch_size - self explanatory
    # features_length - it is the size of a input vector (history_length in CCOD)
    sequence_length, batch_size, fetures_length = x.shape

    core = hk.LSTM(fetures_length)
    initial_state = core.initial_state(batch_size)
    _, lstm_state = hk.dynamic_unroll(core, x, initial_state)
    h_t = lstm_state.hidden

    h_t = hk.nets.MLP([128, 64, NO_ACTIONS])(h_t)

    return jnp.argmax(h_t, axis=1)


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

    def print_environment(observation, action):
        print(f"Sim Time: {'{:.3f}'.format(observation['sim_time'])}\t", end="")
        print(f"History[{len(observation['history'])}]:", end="")
        for p_col in observation['history'][:5]:
            print(" {:.3f}".format(p_col), end="")
        print(f"\tAction: {action}\t", end="")
        print(f"Reward: {'{:.3f}'.format(observation['reward'])}")

    rl = RLib(
        agent_type=agent_type,
        agent_params=agent_params,
        ext_type=IEEE_802_11_ax_CCOD
    )

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
                action = rl.sample(**observation)
                data.act.action = action

                print_environment(observation, action) if verbose else None

        ns3_process.wait()
    finally:
        del exp


if __name__ == '__main__':
    args = ArgumentParser()

    # Python arguments
    args.add_argument('--agent', default="dqn", type=str)
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
        'DQN': DQN
    }
    default_params = {
        'DQN': {
            'q_network': q_network,
            'optimizer': optax.rmsprop(3e-4, decay=0.95, eps=1e-2),
            'discount': 0.95,
            'epsilon_decay': 0.9975
        }
    }

    run(args, args.pop('ns3Path'), args.pop('mempoolKey'), agent_type[agent], default_params[agent], args.pop('pythonSeed'), args['verbose'])
