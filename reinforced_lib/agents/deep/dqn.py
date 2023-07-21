from copy import deepcopy
from functools import partial
from typing import Callable

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
class DQNState(AgentState):
    r"""
    Container for the state of the double Q-learning agent.

    Attributes
    ----------
    params : hk.Params
        Parameters of the main Q-network.
    state : hk.State
        State of the main Q-network.
    params_target : hk.Params
        Parameters of the target Q-network.
    state_target : hk.State
        State of the target Q-network.
    opt_state : optax.OptState
        Optimizer state of the main Q-network.
    replay_buffer : ReplayBuffer
        Experience replay buffer.
    prev_env_state : Array
        Previous environment state.
    epsilon : Scalar
        :math:`\epsilon`-greedy parameter.
    """

    params: hk.Params
    state: hk.State

    params_target: hk.Params
    state_target: hk.State

    opt_state: optax.OptState

    replay_buffer: ReplayBuffer
    prev_env_state: Array
    epsilon: Scalar


class DQN(BaseAgent):
    r"""
    Double Q-learning agent [2]_ with :math:`\epsilon`-greedy exploration and experience replay buffer. The agent
    uses two Q-networks to stabilize the learning process and avoid overestimation of the Q-values. The main Q-network
    is trained to minimize the Bellman error. The target Q-network is updated with a soft update. This agent follows
    the off-policy learning paradigm and is suitable for environments with discrete action spaces.

    Parameters
    ----------
    q_network : hk.TransformedWithState
        Architecture of the Q-networks.
    obs_space_shape : Shape
        Shape of the observation space.
    act_space_size : jnp.int32
        Size of the action space.
    optimizer : optax.GradientTransformation, optional
        Optimizer of the Q-networks. If None, the Adam optimizer with learning rate 1e-3 is used.
    experience_replay_buffer_size : jnp.int32, default=10000
        Size of the experience replay buffer.
    experience_replay_batch_size : jnp.int32, default=64
        Batch size of the samples from the experience replay buffer.
    experience_replay_steps : jnp.int32, default=5
        Number of experience replay steps per update.
    discount : Scalar, default=0.99
        Discount factor. :math:`\gamma = 0.0` means no discount, :math:`\gamma = 1.0` means infinite discount. :math:`0 \leq \gamma \leq 1`
    epsilon : Scalar, default=1.0
        Initial :math:`\epsilon`-greedy parameter. :math:`0 \leq \epsilon \leq 1`.
    epsilon_decay : Scalar, default=0.999
        Epsilon decay factor. :math:`\epsilon_{t+1} = \epsilon_{t} * \epsilon_{decay}`. :math:`0 \leq \epsilon_{decay} \leq 1`.
    epsilon_min : Scalar, default=0.01
        Minimum :math:`\epsilon`-greedy parameter. :math:`0 \leq \epsilon_{min} \leq \epsilon`.
    tau : Scalar, default=0.01
        Soft update factor. :math:`\tau = 0.0` means no soft update, :math:`\tau = 1.0` means hard update. :math:`0 \leq \tau \leq 1`.

    References
    ----------
    .. [2] van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning.
       Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, 2094–2100. Phoenix, Arizona: AAAI Press.
    """

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
            epsilon_decay: Scalar = 0.999,
            epsilon_min: Scalar = 0.001,
            tau: Scalar = 0.01
    ) -> None:

        assert experience_replay_buffer_size > experience_replay_batch_size > 0
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= epsilon <= 1.0
        assert 0.0 <= epsilon_decay <= 1.0
        assert 0.0 <= epsilon_min <= epsilon
        assert 0.0 <= tau <= 1.0

        if optimizer is None:
            optimizer = optax.adam(1e-3)

        self.obs_space_shape = obs_space_shape if jnp.ndim(obs_space_shape) > 0 else (obs_space_shape,)
        self.act_space_size = act_space_size

        er = experience_replay(
            experience_replay_buffer_size,
            experience_replay_batch_size,
            self.obs_space_shape,
            (1,)
        )

        self.init = jax.jit(partial(
            self.init,
            obs_space_shape=self.obs_space_shape,
            q_network=q_network,
            optimizer=optimizer,
            experience_replay=er,
            epsilon=epsilon
        ))
        self.update = jax.jit(partial(
            self.update,
            step_fn=partial(
                gradient_step,
                optimizer=optimizer,
                loss_fn=partial(self.loss_fn, q_network=q_network, discount=discount)
            ),
            experience_replay=er,
            experience_replay_steps=experience_replay_steps,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            tau=tau
        ))
        self.sample = jax.jit(partial(
            self.sample,
            q_network=q_network,
            act_space_size=act_space_size
        ))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'obs_space_shape': gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), jnp.int32)),
            'act_space_size': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'experience_replay_buffer_size': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'experience_replay_batch_size': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'discount': gym.spaces.Box(0.0, 1.0, (1,)),
            'epsilon': gym.spaces.Box(0.0, 1.0, (1,)),
            'epsilon_decay': gym.spaces.Box(0.0, 1.0, (1,)),
            'epsilon_min': gym.spaces.Box(0.0, 1.0, (1,)),
            'tau': gym.spaces.Box(0.0, 1.0, (1,))
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_state': gym.spaces.Box(-jnp.inf, jnp.inf, self.obs_space_shape),
            'action': gym.spaces.Discrete(self.act_space_size),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,)),
            'terminal': gym.spaces.MultiBinary(1)
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_state': gym.spaces.Box(-jnp.inf, jnp.inf, self.obs_space_shape)
        })

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.act_space_size)

    @staticmethod
    def init(
            key: PRNGKey,
            obs_space_shape: Shape,
            q_network: hk.TransformedWithState,
            optimizer: optax.GradientTransformation,
            experience_replay: ExperienceReplay,
            epsilon: Scalar
    ) -> DQNState:
        r"""
        Initializes the Q-networks, optimizer and experience replay buffer with given parameters.
        First state of the environment is assumed to be a tensor of zeros.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        obs_space_shape : Shape
            The shape of the observation space.
        q_network : hk.TransformedWithState
            The Q-network.
        optimizer : optax.GradientTransformation
            The optimizer.
        experience_replay : ExperienceReplay
            The experience replay buffer.
        epsilon : Scalar
            The initial :math:`\epsilon`-greedy parameter.

        Returns
        -------
        DQNState
            Initial state of the double Q-learning agent.
        """

        x_dummy = jnp.empty(obs_space_shape)
        params, state = q_network.init(key, x_dummy)

        opt_state = optimizer.init(params)
        replay_buffer = experience_replay.init()

        return DQNState(
            params=params,
            state=state,
            params_target=deepcopy(params),
            state_target=deepcopy(state),
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=jnp.zeros(obs_space_shape),
            epsilon=epsilon
        )

    @staticmethod
    def loss_fn(
            params: hk.Params,
            key: PRNGKey,
            dqn_state: DQNState,
            batch: tuple,
            non_zero_loss: jnp.bool_,
            q_network: hk.TransformedWithState,
            discount: Scalar
    ) -> tuple[Scalar, hk.State]:
        r"""
        Loss is the mean squared Bellman error :math:`\mathcal{L}(\theta) = \mathbb{E}_{s, a, r, s'} \left[ \left( r +
        \gamma \max_{a'} Q'(s', a') - Q(s, a) \right)^2 \right]` where :math:`s` is the current state, :math:`a` is the
        current action, :math:`r` is the reward, :math:`s'` is the next state, :math:`\gamma` is  the discount factor, 
        :math:`Q(s, a)` is the Q-value of the main Q-network, :math:`Q'(s', a')` is the Q-value of the target
        Q-network. Loss can be calculated on a batch of transitions.

        Parameters
        ----------
        params : hk.Params
            The parameters of the Q-network.
        key : PRNGKey
            A PRNG key used as the random key.
        dqn_state : DQNState
            The state of the double Q-learning agent.
        batch : tuple
            A batch of transitions from the experience replay buffer.
        non_zero_loss : bool
            Flag used to avoid updating the Q-network when the experience replay buffer is not full.
        q_network : hk.TransformedWithState
            The Q-network.
        discount : Scalar
            The discount factor.

        Returns
        -------
        tuple[Scalar, hk.State]
            The loss and the new state of the Q-network.
        """

        states, actions, rewards, terminals, next_states = batch
        q_key, q_target_key = jax.random.split(key)

        q_values, state = q_network.apply(params, dqn_state.state, q_key, states)
        q_values = jnp.take_along_axis(q_values, actions.astype(jnp.int32), axis=-1)

        q_values_target, _ = q_network.apply(dqn_state.params_target, dqn_state.state_target, q_target_key, next_states)
        target = rewards + (1 - terminals) * discount * jnp.max(q_values_target, axis=-1, keepdims=True)

        target = jax.lax.stop_gradient(target)
        loss = optax.l2_loss(q_values, target).mean()

        return loss * non_zero_loss, state

    @staticmethod
    def update(
            state: DQNState,
            key: PRNGKey,
            env_state: Array,
            action: Array,
            reward: Scalar,
            terminal: jnp.bool_,
            step_fn: Callable,
            experience_replay: ExperienceReplay,
            experience_replay_steps: jnp.int32,
            epsilon_decay: Scalar,
            epsilon_min: Scalar,
            tau: Scalar
    ) -> DQNState:
        r"""
        Appends the transition to the experience replay buffer and performs ``experience_replay_steps`` steps.
        Each step consists of sampling a batch of transitions from the experience replay buffer, calculating the loss
        using the ``loss_fn`` function, performing a gradient step on the main Q-network, and soft updating the target
        Q-network. Soft update of the parameters is defined as :math:`\theta_{target} = \tau \theta + (1 - \tau) \theta_{target}`.
        The :math:`\epsilon`-greedy parameter is decayed by ``epsilon_decay``.

        Parameters
        ----------
        state : DQNState
            The current state of the double Q-learning agent.
        key : PRNGKey
            A PRNG key used as the random key.
        env_state : Array
            The current state of the environment.
        action : Array
            The action taken by the agent.
        reward : Scalar
            The reward received by the agent.
        terminal : bool
            Whether the episode has terminated.
        step_fn : Callable
            The function that performs a single gradient step on the Q-network.
        experience_replay : ExperienceReplay
            The experience replay buffer.
        experience_replay_steps : int
            The number of experience replay steps.
        epsilon_decay : Scalar
            The decay rate of the :math:`\epsilon`-greedy parameter.
        epsilon_min : Scalar
            The minimum value of the :math:`\epsilon`-greedy parameter.
        tau : Scalar
            The soft update parameter.

        Returns
        -------
        DQNState
            The updated state of the double Q-learning agent.
        """

        replay_buffer = experience_replay.append(
            state.replay_buffer, state.prev_env_state,
            action, reward, terminal, env_state
        )

        params, net_state, opt_state = state.params, state.state, state.opt_state
        params_target, state_target = state.params_target, state.state_target

        non_zero_loss = experience_replay.is_ready(replay_buffer)

        for _ in range(experience_replay_steps):
            batch_key, network_key, key = jax.random.split(key, 3)
            batch = experience_replay.sample(replay_buffer, batch_key)

            params, net_state, opt_state, _ = step_fn(params, (network_key, state, batch, non_zero_loss), opt_state)
            params_target, state_target = optax.incremental_update(
                (params, net_state), (params_target, state_target), tau)

        return DQNState(
            params=params,
            state=net_state,
            params_target=params_target,
            state_target=state_target,
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=env_state,
            epsilon=jax.lax.max(state.epsilon * epsilon_decay, epsilon_min)
        )

    @staticmethod
    def sample(
            state: DQNState,
            key: PRNGKey,
            env_state: Array,
            q_network: hk.TransformedWithState,
            act_space_size: jnp.int32
    ) -> jnp.int32:
        r"""
        Samples random action with probability :math:`\epsilon` and the greedy action with probability
        :math:`1 - \epsilon` using the main Q-network. The greedy action is the action with the highest Q-value.

        Parameters
        ----------
        state : DQNState
            The state of the double Q-learning agent.
        key : PRNGKey
            A PRNG key used as the random key.
        env_state : Array
            The current state of the environment.
        q_network : hk.TransformedWithState
            The Q-network.
        act_space_size : jnp.int32
            The size of the action space.

        Returns
        -------
        int
            Selected action.
        """

        network_key, epsilon_key, action_key = jax.random.split(key, 3)

        return jax.lax.cond(
            jax.random.uniform(epsilon_key) < state.epsilon,
            lambda: jax.random.choice(action_key, act_space_size),
            lambda: jnp.argmax(q_network.apply(state.params, state.state, network_key, env_state)[0])
        )
