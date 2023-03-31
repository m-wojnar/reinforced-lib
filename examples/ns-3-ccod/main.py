"""
Implementation of CCOD algorithm, according to:
W. Wydmaski and S. Szott, “Contention Window Optimization in IEEE 802.11ax Networks
with Deep Reinforcement Learning” 2021 IEEE Wireless Communications and Networking
Conference (WCNC), 2021. https://doi.org/10.1109/WCNC49053.2021.9417575
"""

from argparse import ArgumentParser
from ctypes import *
from typing import Any, Dict

import haiku as hk
import numpy as np
import optax
from chex import Array

from py_interface import *
from reinforced_lib import RLib
from reinforced_lib.agents.deep import DQN
from reinforced_lib.exts.wifi import IEEE_802_11_CW


# DRL settings, according to Table I from the cited article

INTERACTION_PERIOD = 1e-2
SIMULATION_TIME = 60
MAX_HISTORY_LENGTH = 512
HISTORY_LENGTH = 300

LSTM_HIDDEN_SIZE = 8
REWARD_DISCOUNT = 0.7

DQN_LEARNING_RATE = 4e-4
EPSILON = 0.9
EPSILON_DECAY = 0.99991
EPSILON_MIN = 0.001
SOFT_UPDATE = 4e-3

REPLAY_BUFFER_SIZE = 18000
REPLAY_BUFFER_BATCH_SIZE = 32
REPLAY_BUFFER_STEPS = 1


class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('history', c_float * MAX_HISTORY_LENGTH),
        ('reward', c_float)
    ]


class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ('action', c_uint32)
    ]


memblock_key = 2333
memory_size = 4096
simulation = 'ccod-sim'


@hk.transform_with_state
def q_network(x: Array) -> Array:
    if x.ndim == 2:
        x = x[None, ...]

    core = hk.LSTM(LSTM_HIDDEN_SIZE)
    initial_state = core.initial_state(x.shape[0])
    _, lstm_state = hk.dynamic_unroll(core, x, initial_state, time_major=False)

    h_t = lstm_state.hidden
    return hk.nets.MLP([128, 64, 7])(h_t)


def preprocess(history: Array, history_length: int) -> Array:
    """
    Preprocess the history according to the CCOD algorithm.

    Parameters
    ----------
    history : array_like
        History of the transmission failure probability.
    history_length : int
        Length of the history.

    Returns
    -------
    array_like
        Preprocessed history.
    """

    history = history[:history_length]
    window = history_length // 2
    res = np.empty((4, 2))

    for i, pos in enumerate(range(0, history_length, window // 2)):
        res[i, 0] = np.mean(history[pos:pos + window])
        res[i, 1] = np.std(history[pos:pos + window])

    return np.clip(res, 0, 1)


def run(
        ns3_args: Dict[str, Any],
        ns3_path: str,
        mempool_key: int,
        agent_type: type,
        agent_params: Dict[str, Any],
        seed: int,
        run_id: int
) -> None:
    """
    Run a given number of simulations in the ns-3 simulator [1]_ with the ns3-ai library [2]_.

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
    run_id : int
        Number of the current simulation.

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
        ext_type=IEEE_802_11_CW
    )

    if seed is not None:
        rl.init(seed)
    else:
        rl.load(f'checkpoints/run_{run_id - 1}.pkl.lz4', restore_loggers=False)

    exp = Experiment(mempool_key, memory_size, simulation, ns3_path, debug=False)
    var = Ns3AIRL(memblock_key, Env, Act)

    try:
        ns3_process = exp.run(ns3_args, show_output=True)

        while not var.isFinish():
            with var as data:
                if data is None:
                    break

                observation = {
                    'history': preprocess(data.env.history, ns3_args['historyLength']),
                    'reward': data.env.reward
                }
                data.act.action = rl.sample(**observation)

        ns3_process.wait()
    finally:
        del exp

    rl.save(agent_ids=0, path=f'checkpoints/run_{run_id}.pkl.lz4')


if __name__ == '__main__':
    args = ArgumentParser()

    # Python arguments
    args.add_argument('--agent', default="DQN", type=str)
    args.add_argument('--mempoolKey', default=1234, type=int)
    args.add_argument('--ns3Path', required=True, type=str)
    args.add_argument('--pythonSeed', default=None, type=int)
    args.add_argument('--runId', required=True, type=int)

    # ns3 arguments
    args.add_argument('--agentType', default='discrete', type=str)
    args.add_argument('--CW', default=0, type=int)
    args.add_argument('--dryRun', default=False, action='store_true')
    args.add_argument('--envStepTime', default=INTERACTION_PERIOD, type=float)
    args.add_argument('--historyLength', default=HISTORY_LENGTH, type=int)
    args.add_argument('--nonZeroStart', default=True, action='store_true')
    args.add_argument('--nWifi', default=55, type=int)
    args.add_argument('--rng', default=42, type=int)
    args.add_argument('--scenario', default='convergence', type=str)
    args.add_argument('--simTime', default=SIMULATION_TIME, type=float)
    args.add_argument('--tracing', default=False, action='store_true')
    args.add_argument('--verbose', default=False, action='store_true')

    args = vars(args.parse_args())

    assert args['historyLength'] <= MAX_HISTORY_LENGTH, \
        f"HISTORY_LENGTH={args['historyLength']} exceeded MAX_HISTORY_LENGTH={MAX_HISTORY_LENGTH}, " +\
        f"reduce HISTORY_LENGTH value in 'reinforced_lib/exts/wifi/ieee_802_11_cw.py' file!"

    agent = args.pop('agent')
    agent_type = {
        'DQN': DQN
    }
    default_params = {
        'DQN': {
            'q_network':                        q_network,
            'optimizer':                        optax.sgd(DQN_LEARNING_RATE),
            'experience_replay_buffer_size':    REPLAY_BUFFER_SIZE,
            'experience_replay_batch_size':     REPLAY_BUFFER_BATCH_SIZE,
            'experience_replay_steps':          REPLAY_BUFFER_STEPS,
            'discount':                         REWARD_DISCOUNT,
            'epsilon':                          EPSILON,
            'epsilon_decay':                    EPSILON_DECAY,
            'epsilon_min':                      EPSILON_MIN,
            'tau':                              SOFT_UPDATE
        }
    }

    run(args, args.pop('ns3Path'), args.pop('mempoolKey'), agent_type[agent],
        default_params[agent], args.pop('pythonSeed'), args.pop('runId'))
