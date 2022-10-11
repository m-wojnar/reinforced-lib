import game
import gym

gym.logger.set_level(40)


SEED = 42

if __name__ == "__main__":

    env = gym.make('BlackjackEnv')
    state, _ = env.reset(seed=SEED)
    env.render()

    done = False
    while not done:

        act = int(input("Type 1 for hit or 0 for stick: "))
        print(f"action = {act}")
        state, reward, done, _ = env.step(act)
        env.render()