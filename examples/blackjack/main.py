import blackjack
import gym
from reinforced_lib import RLib
from reinforced_lib.agents import GradientBandit

gym.logger.set_level(40)


SEED = 42


# Create blackjack gym environment
env = gym.make('BlackjackEnv-v0')
state, _ = env.reset(seed=SEED)

# Create contexts as possible points accumulated in game
contexts = range(env.observation_space['current_sum'].start, env.observation_space['current_sum'].n)
agent_params={"n_arms": 2, "lr": 0.005}

# Create and initialize rl agent for each points context
rl_context = {ctx: RLib(agent_type=GradientBandit, agent_params=agent_params) for ctx in contexts}
for ctx in contexts:
    rl_context[ctx].init(SEED)
