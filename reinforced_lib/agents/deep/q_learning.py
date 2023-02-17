from copy import deepcopy
from functools import partial
from typing import Callable, Tuple

import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from chex import dataclass, Array, PRNGKey, Scalar, Shape

from reinforced_lib.agents import BaseAgent, AgentState
from reinforced_lib.utils.experience_replay import experience_replay, ExperienceReplay, ReplayBuffer
from reinforced_lib.utils.jax_utils import gradient_step


@dataclass
class QLearningState(AgentState):
    """
    Container for the state of the deep Q-learning agent.

    Attributes
    ----------
    params : hk.Params
        Parameters of the Q-network.
    state : hk.State
        State of the Q-network.
    opt_state : optax.OptState
        Optimizer state.
    replay_buffer : ReplayBuffer
        Experience replay buffer.
    prev_env_state : Array
        Previous environment state.
    epsilon : Scalar
        Epsilon-greedy parameter.
    """

    params: hk.Params
    state: hk.State
    opt_state: optax.OptState

    replay_buffer: ReplayBuffer
    prev_env_state: Array
    epsilon: Scalar


class QLearning(BaseAgent):

    def __init__(
            self,
            q_network: hk.TransformedWithState,
            obs_space_shape: Shape,
            act_space_size: jnp.int32,
            optimizer: optax.GradientTransformation = None,
            experience_replay_buffer_size: jnp.int32 = 10000,
            experience_replay_batch_size: jnp.int32 = 64,
            experience_replay_steps: jnp.int32 = 5,
            discount: Scalar = 0.99,
            epsilon: Scalar = 1.0,
            epsilon_decay: Scalar = 0.999
    ) -> None:

        assert experience_replay_buffer_size > experience_replay_batch_size > 0
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= epsilon <= 1.0
        assert 0.0 <= epsilon_decay <= 1.0

        if optimizer is None:
            optimizer = optax.adam(1e-3)

        self.obs_space_shape = obs_space_shape if jnp.ndim(obs_space_shape) > 0 else (obs_space_shape,)
        self.act_space_shape = (act_space_size,)

        er = experience_replay(
            experience_replay_buffer_size,
            experience_replay_batch_size,
            self.obs_space_shape,
            self.act_space_shape
        )

        self.init = jax.jit(partial(
            self.init,
            obs_space_shape=self.obs_space_shape,
            q_network=q_network,
            optimizer=optimizer,
            experience_replay=er,
            epsilon=epsilon
        ))
        self.update = partial(
            self.update,
            q_network=q_network,
            step_fn=jax.jit(partial(
                gradient_step,
                optimizer=optimizer,
                loss_fn=partial(self._loss_fn, q_network=q_network, discount=discount)
            )),
            experience_replay=er,
            experience_replay_steps=experience_replay_steps,
            discount=discount,
            epsilon_decay=epsilon_decay
        )
        self.sample = jax.jit(partial(
            self.sample,
            q_network=q_network,
            act_space_size=act_space_size
        ))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'q_network': hk.TransformedWithState,
            'obs_space_shape': Shape,
            'act_space_size': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'optimizer': optax.GradientTransformation,
            'experience_replay_buffer_size': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'experience_replay_batch_size': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'discount': gym.spaces.Box(0.0, 1.0, (1,)),
            'epsilon': gym.spaces.Box(0.0, 1.0, (1,)),
            'epsilon_decay': gym.spaces.Box(0.0, 1.0, (1,))
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_state': gym.spaces.Box(-jnp.inf, jnp.inf, self.obs_space_shape),
            'action': gym.spaces.Box(-jnp.inf, jnp.inf, self.act_space_shape),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'terminal': gym.spaces.Discrete(2)
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_state': gym.spaces.Box(-jnp.inf, jnp.inf, self.obs_space_shape)
        })

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-jnp.inf, jnp.inf, self.act_space_shape)

    @staticmethod
    def init(
            key: PRNGKey,
            obs_space_shape: Shape,
            q_network: hk.TransformedWithState,
            optimizer: optax.GradientTransformation,
            experience_replay: ExperienceReplay,
            epsilon: Scalar
    ) -> QLearningState:

        x_dummy = jnp.empty(obs_space_shape)
        params, state = q_network.init(key, x_dummy)

        opt_state = optimizer.init(params)
        replay_buffer = experience_replay.init()

        return QLearningState(
            params=params,
            state=state,
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=jnp.zeros(obs_space_shape),
            epsilon=epsilon
        )

    @staticmethod
    def _loss_fn(
            params: hk.Params,
            key: PRNGKey,
            state: hk.State,
            params_target: hk.Params,
            state_target: hk.State,
            batch: Tuple,
            q_network: hk.TransformedWithState,
            discount: Scalar
    ) -> Tuple[Scalar, hk.State]:
        states, actions, rewards, terminals, next_states = batch
        q_key, q_target_key = jax.random.split(key)

        q_values, state = q_network.apply(params, state, q_key, states)
        q_values = jnp.take_along_axis(q_values, actions.astype(jnp.int32), axis=-1)

        q_values_target, _ = q_network.apply(params_target, state_target, q_target_key, next_states)
        target = rewards + (1 - terminals) * discount * jnp.argmax(q_values_target, axis=-1)

        target = jax.lax.stop_gradient(target)
        loss = jnp.square(target - jnp.squeeze(q_values)).mean()

        return loss, state

    @staticmethod
    def update(
            state: QLearningState,
            key: PRNGKey,
            env_state: Array,
            action: Array,
            reward: Scalar,
            terminal: jnp.bool_,
            q_network: hk.TransformedWithState,
            step_fn: Callable,
            experience_replay: ExperienceReplay,
            experience_replay_steps: jnp.int32,
            discount: Scalar,
            epsilon_decay: Scalar
    ) -> QLearningState:

        replay_buffer = experience_replay.append(
            state.replay_buffer, state.prev_env_state,
            action, reward, terminal, env_state
        )

        params, network_state, opt_state = state.params, state.state, state.opt_state

        if experience_replay.is_ready(replay_buffer):
            params_target = deepcopy(state.params)
            state_target = deepcopy(state.state)

            for _ in range(experience_replay_steps):
                batch_key, network_key, key = jax.random.split(key, 3)
                batch = experience_replay.sample(replay_buffer, batch_key)

                params, network_state, opt_state, loss = step_fn(
                    params,
                    (key, state.state, params_target, state_target, batch),
                    opt_state
                )

        return QLearningState(
            params=params,
            state=network_state,
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=env_state,
            epsilon=state.epsilon * epsilon_decay
        )

    @staticmethod
    def sample(
            state: QLearningState,
            key: PRNGKey,
            env_state: Array,
            q_network: hk.TransformedWithState,
            act_space_size: jnp.int32
    ) -> Array:

        epsilon_key, action_key, network_key = jax.random.split(key, 3)
        action = jnp.argmax(q_network.apply(state.params, state.state, network_key, env_state)[0])

        return jax.lax.cond(
            jax.random.uniform(epsilon_key) < state.epsilon,
            lambda: jax.random.choice(action_key, act_space_size),
            lambda: action
        )
