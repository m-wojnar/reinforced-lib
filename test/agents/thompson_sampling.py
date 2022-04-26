import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from agents.thompson_sampling import thompson_sampling


if __name__ == '__main__':
    # environment characteristics
    arms_probs = jnp.array([0.1, 0.2, 0.3, 0.8, 0.3])
    context = jnp.array([5.0, 5.0, 2.0, 2.0, 5.0])

    # agent setup
    agent = thompson_sampling(context)
    agent = agent._replace(
        sample=jax.jit(agent.sample),
        update=jax.jit(agent.update)
    )
    state = agent.init()

    # helper variables
    key = jax.random.PRNGKey(42)
    time = jnp.linspace(0, 20, 2000)
    actions = []
    a = 0

    for t in time:
        # pull selected arm
        key, random_key, sample_key = jax.random.split(key, 3)
        r = jax.random.uniform(random_key) < arms_probs[a]

        # update state and sample
        state = agent.update(state, a, r, t)
        a, state = agent.sample(state, sample_key, t)

        # save selected action
        actions.append(a)

    # print agent state
    print(f'TS approximate arms probabilities: {state.alpha / (state.alpha + state.beta)}')

    # action vs time plot
    plt.figure(figsize=(8, 5), dpi=200)
    plt.scatter(time, actions, s=2)
    plt.ylim((-0.5, 4.5))
    plt.title('ThompsonSampling agent [decay = 1.0]')
    plt.xlabel('Time')
    plt.ylabel('Action')
    plt.show()
