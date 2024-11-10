import jax
import jax.numpy as jnp


from reinforced_lib.agents.mab import ThompsonSampling
from reinforced_lib.agents.mab.masked import Masked

if __name__ == '__main__':
    # environment characteristics
    arms_probs = jnp.array([0.1, 0.2, 0.3, 0.8, 0.3])
    context = jnp.array([5.0, 5.0, 2.0, 2.0, 5.0])

    # agent setup
    decay = 1.0
    agent = ThompsonSampling(len(context), decay)

    mask = jnp.asarray([0,0,0,0,1], dtype=jnp.bool)

    agent = Masked(agent,mask)

    k = jax.random.key(4)
    state = agent.init(k)
    # print observation and action spaces
    # print(agent.update_observation_space)
    # print(agent.sample_observation_space)
    # print(agent.action_space)

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
        a = agent.sample(state, sample_key, context)

        # save selected action
        actions.append(a)

assert 4 not in actions