import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from reinforced_lib.agents import ThompsonSampling


if __name__ == '__main__':
    # environment characteristics
    arms_probs = jnp.array([0.1, 0.2, 0.3, 0.8, 0.3])
    context = jnp.array([5.0, 5.0, 2.0, 2.0, 5.0])

    # agent setup
    decay = 1.0
    agent = ThompsonSampling(len(context), decay)
    state = agent.init()

    # print observation and action spaces
    print(agent.update_observation_space)
    print(agent.sample_observation_space)
    print(agent.action_space)

    # helper variables
    key = jax.random.PRNGKey(42)
    time = jnp.linspace(0, 20, 2000)
    actions = []
    a = 0

    for t in time:
        # pull selected arm
        key, random_key, update_key, sample_key = jax.random.split(key, 4)
        r = jax.random.uniform(random_key) < arms_probs[a]

        # update state and sample
        state = agent.update(state, update_key, a, r, 1 - r, t)
        state, a = agent.sample(state, sample_key, t, context)

        # save selected action
        actions.append(a)

    # print agent state
    print(f'TS approximate arms probabilities: {state.alpha / (state.alpha + state.beta)}')

    # action vs time plot
    plt.figure(figsize=(8, 5), dpi=200)
    plt.scatter(time, actions, s=2)
    plt.ylim((-0.5, 4.5))
    plt.title(f'ThompsonSampling agent [decay = {decay}]')
    plt.xlabel('Time')
    plt.ylabel('Action')
    plt.show()
