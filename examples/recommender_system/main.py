import numpy as np
import matplotlib.pyplot as plt
import gym
import recommender_system_env
from reinforced_lib import RLib
from reinforced_lib.agents import EGreedy
from reinforced_lib.logs import PlotsLogger
from reinforced_lib.logs.base_logger import SourceType

gym.logger.set_level(40)


SEED = 47
N_EPISODES = 300


# Create users preferences
preferences = {
    "astronomy": 0.71,      # 0
    "nature": 0.6,          # 1
    "cooking": 0.6,         # 2
    "games": 0.2,           # 3
    "music": 0.92,          # 4 TOP
    "sports": 0.4,          # 5
    "technology": 0.67      # 6
}

# Create and reset the environment which will simulate users behavior
env = gym.make("RecommenderSystemEnv-v1", preferences=preferences)
_ = env.reset(seed=SEED)

# Wrap everything under RLib object with designated agent
agent_params = {"n_arms": len(preferences), "e": 0.25}
rl = RLib(
    agent_type=EGreedy,
    agent_params=agent_params,
    no_ext_mode=True,
    # loggers_type=PlotsLogger,
    # loggers_sources=[('action', SourceType.METRIC)]
)
rl.init(SEED)

# Create data structures to store the history
actions_trace = np.empty(N_EPISODES)
rewards_trace = np.empty(N_EPISODES)

# Loop through each episode and update prior knowledge
act = env.action_space.sample()
_, reward, _, _, _ = env.step(act)
actions_trace[0] = act
rewards_trace[0] = reward
for i in range(1, N_EPISODES):

    act = rl.sample(update_observations={"action": act, "reward": reward})
    _, reward, *_ = env.step(act)
    actions_trace[i] = act
    rewards_trace[i] = reward
    

# Plot the trace of undertaken actions
plt.scatter(np.arange(N_EPISODES), actions_trace, marker=".")
plt.yticks(ticks=range(len(preferences)), labels=list(preferences.keys()))
plt.xlabel("time in actions")
plt.show()
