import numpy as np
import matplotlib.pyplot as plt
import warnings
import gym
import env

from reinforced_lib import RLib
from reinforced_lib.agents import EGreedy
from reinforced_lib.logs import PlotsLogger
from reinforced_lib.logs import SourceType

gym.logger.set_level(40)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def run(episodes: int, seed: int) -> None:

    # Create users preferences
    preferences = {
        "astronomy": 0.71,      # 0
        "nature": 0.6,          # 1
        "cooking": 0.6,         # 2
        "games": 0.2,           # 3
        "music": 0.92,          # 4 Highest expected reward
        "sports": 0.4,          # 5
        "technology": 0.67      # 6
    }

    # Create and reset the environment which will simulate users behavior
    env = gym.make("RecommenderSystemEnv-v1", preferences=preferences)
    _ = env.reset(seed=seed)

    # Wrap everything under RLib object with designated agent
    agent_params = {"n_arms": len(preferences), "e": 0.25}
    rl = RLib(
        agent_type=EGreedy,
        agent_params=agent_params,
        no_ext_mode=True,
        loggers_type=PlotsLogger,
        loggers_sources=[('action', SourceType.METRIC)],
        loggers_params={'scatter': True}
    )
    rl.init(seed)

    # Create data structures to store rewards
    total_reward = 0
    cumulative_reward = np.empty(episodes)

    # Loop through each episode and update prior knowledge
    act = env.action_space.sample()
    _, reward, _, _, _ = env.step(act)
    total_reward += reward
    cumulative_reward[0] = total_reward
    for i in range(1, episodes):

        act = rl.sample(update_observations={"action": act, "reward": reward})
        _, reward, *_ = env.step(act)
        total_reward += reward
        cumulative_reward[i] = total_reward
    
    # Plot the cumulative reward
    plt.plot(np.arange(episodes), cumulative_reward)
    plt.xlabel("steps")
    plt.ylabel("total reward")
    plt.title("reward-metric")
    plt.show()


if __name__ == "__main__":

    run(300, 47)
