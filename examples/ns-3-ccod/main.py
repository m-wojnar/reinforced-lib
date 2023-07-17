"""
Implementation of CCOD algorithm, according to:
W. Wydmaski and S. Szott, “Contention Window Optimization in IEEE 802.11ax Networks
with Deep Reinforcement Learning” 2021 IEEE Wireless Communications and Networking
Conference (WCNC), 2021. https://doi.org/10.1109/WCNC49053.2021.9417575
"""

from argparse import ArgumentParser
from ctypes import *

import haiku as hk
import jax.numpy as jnp
import optax
from chex import Array

from ext import IEEE_802_11_CCOD
from py_interface import *

from reinforced_lib import RLib
from reinforced_lib.agents.deep import DQN, DDPG
from reinforced_lib.logs import SourceType, TensorboardLogger


# DRL settings, according to the cited article and its source code

INTERACTION_PERIOD = 1e-2
SIMULATION_TIME = 60
MAX_HISTORY_LENGTH = IEEE_802_11_CCOD.max_history_length
HISTORY_LENGTH = 300
THR_SCALE = 5 * 150 * INTERACTION_PERIOD * 10

DQN_LEARNING_RATE = 4e-4
DQN_EPSILON = 0.9
DQN_EPSILON_DECAY = 0.99991
DQN_EPSILON_MIN = 0.001

DDPG_Q_LEARNING_RATE = 4e-3
DDPG_A_LEARNING_RATE = 4e-4
DDPG_NOISE = 4.0
DDPG_NOISE_DECAY = 0.99994
DDPG_NOISE_MIN = 0.001

REWARD_DISCOUNT = 0.7
LSTM_HIDDEN_SIZE = 8
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
        ('action', c_float)
    ]


memblock_key = 2333
memory_size = 4096
simulation = 'ccod-sim'


def add_batch_dim(x: Array, base_ndims: jnp.int32) -> Array:
    if x.ndim == base_ndims and base_ndims > 1:
        return x[None, ...]
    elif x.ndim == base_ndims and base_ndims == 1:
        return x[..., None]
    else:
        return x


def apply_lstm(x: Array, hidden_size: jnp.int32) -> Array:
    core = hk.LSTM(hidden_size)
    initial_state = core.initial_state(x.shape[0])
    _, lstm_state = hk.dynamic_unroll(core, x, initial_state, time_major=False)
    return lstm_state.hidden


@hk.transform_with_state
def dqn_network(x: Array) -> Array:
    x = add_batch_dim(x, base_ndims=2)
    x = apply_lstm(x, LSTM_HIDDEN_SIZE)
    return hk.nets.MLP([128, 64, 7])(x)


@hk.transform_with_state
def ddpg_q_network(s: Array, a: Array) -> Array:
    s = add_batch_dim(s, base_ndims=2)
    s = apply_lstm(s, LSTM_HIDDEN_SIZE)
    a = add_batch_dim(a, base_ndims=1)
    x = jnp.concatenate([s, a], axis=1)
    return hk.nets.MLP([128, 64, 1])(x)


@hk.transform_with_state
def ddpg_a_network(s: Array) -> Array:
    s = add_batch_dim(s, base_ndims=2)
    s = apply_lstm(s, LSTM_HIDDEN_SIZE)
    return hk.nets.MLP([128, 64, 1])(s).squeeze()


def run(
        ns3_args: dict[str, any],
        ns3_path: str,
        mempool_key: int,
        agent_type: type,
        agent_params: dict[str, any],
        rlib_args: dict[str, any]
) -> None:
    """
    Run a CCOD simulation in the ns-3 simulator [1]_ with the ns3-ai library [2]_.

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
    rlib_args : dict
        Arguments used by Reinforced-lib. If `load_path` is not None, the agent will be loaded from the
        specified path. If `is_training` is True, the agent will be updated during the interaction with
        the simulator. If `save_path` is not empty and `is_training` is True, the agent will be saved to
        the specified path. If `csv_path` is not empty, the results will be saved to the specified path.

    References
    ----------
    .. [1] The ns-3 network simulator. http://www.nsnam.org/.
    .. [2] Yin, H., Liu, P., Liu, K., Cao, L., Zhang, L., Gao, Y., & Hei, X. (2020). Ns3-Ai: Fostering Artificial
       Intelligence Algorithms for Networking Research. In Proceedings of the 2020 Workshop on Ns-3 (pp. 57–64).
       Association for Computing Machinery.
    """

    csv_path = rlib_args['csv_path']
    csv_file = open(csv_path, 'w') if csv_path else None

    if not rlib_args['load_path']:
        rl = RLib(
            agent_type=agent_type,
            agent_params=agent_params,
            ext_type=IEEE_802_11_CCOD,
            ext_params={'history_length': ns3_args['historyLength']},
            logger_types=TensorboardLogger,
            logger_sources=[('action', SourceType.METRIC), ('reward', SourceType.METRIC)]
        )
        rl.init(rlib_args['seed'])
    else:
        rl = RLib.load(rlib_args['load_path'])

    exp = Experiment(mempool_key, memory_size, simulation, ns3_path, debug=False)
    var = Ns3AIRL(memblock_key, Env, Act)

    try:
        ns3_process = exp.run(ns3_args, show_output=True)
        step = 0

        while not var.isFinish():
            with var as data:
                if data is None:
                    break

                observation = {
                    'history': data.env.history,
                    'reward': data.env.reward
                }
                data.act.action = rl.sample(**observation, is_training=rlib_args['is_training'])

                csv_file.write(
                    f"{agent_type.__name__},{ns3_args['scenario']},{ns3_args['nWifi']},{ns3_args['RngRun']},"
                    f"{step * INTERACTION_PERIOD},{observation['reward'] * THR_SCALE}\n"
                ) if csv_file else None
                step += 1

        ns3_process.wait()
    finally:
        del exp
        csv_file.close() if csv_file else None

    if rlib_args['is_training'] and rlib_args['save_path']:
        rl.save(agent_ids=0, path=rlib_args['save_path'])


if __name__ == '__main__':
    args = ArgumentParser()

    # Python arguments
    args.add_argument('--agent', default='DQN', type=str)
    args.add_argument('--loadPath', default='', type=str)
    args.add_argument('--mempoolKey', default=1234, type=int)
    args.add_argument('--ns3Path', required=True, type=str)
    args.add_argument('--pythonSeed', default=42, type=int)
    args.add_argument('--sampleOnly', default=False, action='store_true')
    args.add_argument('--savePath', default='', type=str)
    args.add_argument('--csvPath', default='', type=str)

    # ns3 arguments
    args.add_argument('--agentType', default='discrete', type=str)
    args.add_argument('--CW', default=0, type=int)
    args.add_argument('--dryRun', default=False, action='store_true')
    args.add_argument('--envStepTime', default=INTERACTION_PERIOD, type=float)
    args.add_argument('--historyLength', default=HISTORY_LENGTH, type=int)
    args.add_argument('--nonZeroStart', default=True, action='store_true')
    args.add_argument('--nWifi', default=55, type=int)
    args.add_argument('--scenario', default='convergence', type=str)
    args.add_argument('--seed', default=42, type=int)
    args.add_argument('--simTime', default=SIMULATION_TIME, type=float)
    args.add_argument('--tracing', default=False, action='store_true')
    args.add_argument('--verbose', default=False, action='store_true')

    args = vars(args.parse_args())

    assert args['historyLength'] <= MAX_HISTORY_LENGTH, \
        f"HISTORY_LENGTH={args['historyLength']} exceeded MAX_HISTORY_LENGTH={MAX_HISTORY_LENGTH}!"

    args['RngRun'] = args.pop('seed')
    agent = args.pop('agent')

    agent_type = {
        'DQN': DQN,
        'DDPG': DDPG
    }
    default_params = {
        'DQN': {
            'q_network': dqn_network,
            'optimizer': optax.adam(DQN_LEARNING_RATE),
            'experience_replay_buffer_size': REPLAY_BUFFER_SIZE,
            'experience_replay_batch_size': REPLAY_BUFFER_BATCH_SIZE,
            'experience_replay_steps': REPLAY_BUFFER_STEPS,
            'discount': REWARD_DISCOUNT,
            'epsilon': DQN_EPSILON,
            'epsilon_decay': DQN_EPSILON_DECAY,
            'epsilon_min': DQN_EPSILON_MIN,
            'tau': SOFT_UPDATE
        },
        'DDPG': {
            'a_network': ddpg_a_network,
            'a_optimizer': optax.adam(DDPG_A_LEARNING_RATE),
            'q_network': ddpg_q_network,
            'q_optimizer': optax.adam(DDPG_Q_LEARNING_RATE),
            'experience_replay_buffer_size': REPLAY_BUFFER_SIZE,
            'experience_replay_batch_size': REPLAY_BUFFER_BATCH_SIZE,
            'experience_replay_steps': REPLAY_BUFFER_STEPS,
            'discount': REWARD_DISCOUNT,
            'noise': DDPG_NOISE,
            'noise_decay': DDPG_NOISE_DECAY,
            'noise_min': DDPG_NOISE_MIN,
            'tau': SOFT_UPDATE
        }
    }

    rlib_args = {
        'seed': args.pop('pythonSeed'),
        'is_training': not args.pop('sampleOnly'),
        'load_path': args.pop('loadPath'),
        'save_path': args.pop('savePath'),
        'csv_path': args.pop('csvPath')
    }

    run(
        args, 
        args.pop('ns3Path'), 
        args.pop('mempoolKey'), 
        agent_type[agent],
        default_params[agent], 
        rlib_args
    )
