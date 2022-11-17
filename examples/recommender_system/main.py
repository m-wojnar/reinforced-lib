import gym
gym.logger.set_level(40)

import env
from ext import RecommenderSystemExt
from reinforced_lib import RLib
from reinforced_lib.agents import EGreedy
from reinforced_lib.logs import PlotsLogger, SourceType


def run(episodes: int, seed: int) -> None:

    # Construct the extension
    ext = RecommenderSystemExt()

    # Create and reset the environment which will simulate users behavior
    env = gym.make("RecommenderSystemEnv-v1", preferences=ext.preferences)
    _ = env.reset(seed=seed)

    # Wrap everything under RLib object with designated agent
    rl = RLib(
        agent_type=EGreedy,
        agent_params={'e': 0.25},
        ext_type=RecommenderSystemExt,
        loggers_type=PlotsLogger,
        loggers_sources=[('action', SourceType.METRIC), ('cumulative', SourceType.METRIC)],
        loggers_params={'scatter': True}
    )
    rl.init(seed)

    # Loop through each episode and update prior knowledge
    act = env.action_space.sample()
    _, reward, *_ = env.step(act)

    for i in range(1, episodes):
        act = rl.sample(action=act, reward=reward, time=i)
        _, reward, *_ = env.step(act)


if __name__ == "__main__":
    run(300, 47)
