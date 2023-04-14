from argparse import ArgumentParser

from reinforced_lib import RLib
from reinforced_lib.agents.deep.ddpg import DDPGState
from reinforced_lib.agents.deep.dqn import DQNState


if __name__ == '__main__':
    """
    Creates a test checkpoint for the DRL agent in the ns-3-ccod example. The test checkpoint
    contains the same parameters as the original checkpoint, but the training parameters
    and the replay buffer are set to None. The additional parameters (e.g. epsilon) are
    set such that the agent will always choose the action according to the policy network.
    """

    args = ArgumentParser()
    args.add_argument('--loadPath', required=True, type=str)
    args.add_argument('--savePath', required=True, type=str)
    args.add_argument('--agent', required=True, type=str)
    args = args.parse_args()

    rl = RLib.load(args.loadPath)

    if args.agent == 'DQN':
        rl._agent_containers[0].state = DQNState(
            params=rl._agent_containers[0].state.params,
            state=rl._agent_containers[0].state.state,
            params_target=None,
            state_target=None,
            opt_state=None,
            replay_buffer=None,
            prev_env_state=None,
            epsilon=0
        )
    elif args.agent == 'DDPG':
        rl._agent_containers[0].state = DDPGState(
            q_params=None,
            q_state=None,
            q_params_target=None,
            q_state_target=None,
            q_opt_state=None,
            a_params=rl._agent_containers[0].state.a_params,
            a_state=rl._agent_containers[0].state.a_state,
            a_params_target=None,
            a_state_target=None,
            a_opt_state=None,
            replay_buffer=None,
            prev_env_state=None,
            noise=0
        )

    rl.save(agent_ids=0, path=args.savePath)
