import jax
import jax.numpy as jnp
import distrax as jd
import matplotlib.pyplot as plt

from reinforced_lib.agents import EGreedy

if __name__ == '__main__':
    # environment characteristics
    mean_r = jnp.array([5.0, 8.3, 8.0, 3.0])
    std_r = jnp.array([1.0, 1.5, 2.5, 7.0])
    r_dist = jd.Normal(mean_r, std_r)

    # agent setup
    agent = EGreedy(len(mean_r), 0.1, optimistic_start=0.0)
    state = agent.init()

    # print observation and action spaces
    print(agent.update_observation_space)
    print(agent.sample_observation_space)
    print(agent.action_space)

    # helper variables
    key = jax.random.PRNGKey(43)
    time = jnp.linspace(0, 20, 1000)

    actions = []
    state, a = agent.sample(state, key)
    # print(state)
    for t in time:
        # pull selected arm
        key, update_key, sample_key = jax.random.split(key, 3)
        reward = r_dist.sample(seed=key)[a]

        # update state and sample
        state = agent.update(state, update_key, a, reward)
        state, a = agent.sample(state, sample_key)

        # save selected action
        # print('\n', state)
        actions.append(a)
    
    print(f"Best arm so far -> {jnp.argmax(state.q)}")

    # action vs time plot
    plt.figure(figsize=(8, 5), dpi=200)
    plt.scatter(time, actions, s=2)
    plt.ylim((-0.5, 4.5))
    plt.title(f'e-greedy agent, arms={mean_r}')
    plt.xlabel('Time')
    plt.ylabel('Action')
    plt.show()
    
