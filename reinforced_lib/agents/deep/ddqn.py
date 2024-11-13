from copy import deepcopy
from functools import partial
from typing import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from chex import dataclass, Array, PRNGKey, Scalar, Shape
from flax import linen as nn

from reinforced_lib.agents import BaseAgent, AgentState
from reinforced_lib.utils.experience_replay import experience_replay, ExperienceReplay, ReplayBuffer
from reinforced_lib.utils.jax_utils import forward, gradient_step, init


@dataclass
class DDQNState(AgentState):
    r"""
    Container for the state of the double deep Q-learning agent.

    Attributes
    ----------
    params : dict
        Parameters of the main Q-network.
    net_state : dict
        State of the main Q-network.
    params_target : dict
        Parameters of the target Q-network.
    net_state_target : dict
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

    params: dict
    net_state: dict

    params_target: dict
    net_state_target: dict

    opt_state: optax.OptState

    replay_buffer: ReplayBuffer
    prev_env_state: Array
    epsilon: Scalar


class DDQN(BaseAgent):
    r"""
    Double deep Q-learning agent [2]_ with :math:`\epsilon`-greedy exploration and experience replay buffer. The agent
    uses two Q-networks to stabilize the learning process and avoid overestimation of the Q-values. The main Q-network
    is trained to minimize the Bellman error. The target Q-network is updated with a soft update. This agent follows
    the off-policy learning paradigm and is suitable for environments with discrete action spaces.

    Parameters
    ----------
    q_network : nn.Module
        Architecture of the Q-networks.
    obs_space_shape : Shape
        Shape of the observation space.
    act_space_size : int
        Size of the action space.
    optimizer : optax.GradientTransformation, optional
        Optimizer of the Q-networks. If None, the Adam optimizer with learning rate 1e-3 is used.
    experience_replay_buffer_size : int, default=10000
        Size of the experience replay buffer.
    experience_replay_batch_size : int, default=64
        Batch size of the samples from the experience replay buffer.
    experience_replay_steps : int, default=5
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
            q_network: nn.Module,
            obs_space_shape: Shape,
            act_space_size: int,
            optimizer: optax.GradientTransformation = None,
            experience_replay_buffer_size: int = 10000,
            experience_replay_batch_size: int = 64,
            experience_replay_steps: int = 5,
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
            er=er,
            epsilon=epsilon
        ))
        self.update = jax.jit(partial(
            self.update,
            step_fn=partial(
                gradient_step,
                optimizer=optimizer,
                loss_fn=partial(self.loss_fn, q_network=q_network, discount=discount)
            ),
            er=er,
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
            'obs_space_shape': gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), int)),
            'act_space_size': gym.spaces.Box(1, jnp.inf, (1,), int),
            'experience_replay_buffer_size': gym.spaces.Box(1, jnp.inf, (1,), int),
            'experience_replay_batch_size': gym.spaces.Box(1, jnp.inf, (1,), int),
            'discount': gym.spaces.Box(0.0, 1.0, (1,), float),
            'epsilon': gym.spaces.Box(0.0, 1.0, (1,), float),
            'epsilon_decay': gym.spaces.Box(0.0, 1.0, (1,), float),
            'epsilon_min': gym.spaces.Box(0.0, 1.0, (1,), float),
            'tau': gym.spaces.Box(0.0, 1.0, (1,), float)
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_state': gym.spaces.Box(-jnp.inf, jnp.inf, self.obs_space_shape, float),
            'action': gym.spaces.Discrete(self.act_space_size),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float),
            'terminal': gym.spaces.MultiBinary(1)
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_state': gym.spaces.Box(-jnp.inf, jnp.inf, self.obs_space_shape, float)
        })

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.act_space_size)

    @staticmethod
    def init(
            key: PRNGKey,
            obs_space_shape: Shape,
            q_network: nn.Module,
            optimizer: optax.GradientTransformation,
            er: ExperienceReplay,
            epsilon: Scalar
    ) -> DDQNState:
        r"""
        Initializes the Q-networks, optimizer and experience replay buffer with given parameters.
        The first state of the environment is assumed to be a tensor of zeros.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        obs_space_shape : Shape
            The shape of the observation space.
        q_network : nn.Module
            The Q-network.
        optimizer : optax.GradientTransformation
            The optimizer.
        er : ExperienceReplay
            The experience replay buffer.
        epsilon : Scalar
            The initial :math:`\epsilon`-greedy parameter.

        Returns
        -------
        DDQNState
            Initial state of the double Q-learning agent.
        """

        x_dummy = jnp.empty(obs_space_shape)
        params, net_state = init(q_network, key, x_dummy)

        opt_state = optimizer.init(params)
        replay_buffer = er.init()

        return DDQNState(
            params=params,
            net_state=net_state,
            params_target=deepcopy(params),
            net_state_target=deepcopy(net_state),
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=jnp.zeros(obs_space_shape),
            epsilon=epsilon
        )

    @staticmethod
    def loss_fn(
            params: dict,
            key: PRNGKey,
            state: DDQNState,
            batch: tuple,
            q_network: nn.Module,
            discount: Scalar
    ) -> tuple[Scalar, dict]:
        r"""
        Loss is the mean squared Bellman error :math:`\mathcal{L}(\theta) = \mathbb{E}_{s, a, r, s'} \left[ \left( r +
        \gamma \max_{a'} Q'(s', a') - Q(s, a) \right)^2 \right]` where :math:`s` is the current state, :math:`a` is the
        current action, :math:`r` is the reward, :math:`s'` is the next state, :math:`\gamma` is  the discount factor, 
        :math:`Q(s, a)` is the Q-value of the main Q-network, :math:`Q'(s', a')` is the Q-value of the target
        Q-network. Loss can be calculated on a batch of transitions.

        Parameters
        ----------
        params : dict
            The parameters of the Q-network.
        key : PRNGKey
            A PRNG key used as the random key.
        state : DDQNState
            The state of the double deep Q-learning agent.
        batch : tuple
            A batch of transitions from the experience replay buffer.
        q_network : nn.Module
            The Q-network.
        discount : Scalar
            The discount factor.

        Returns
        -------
        tuple[Scalar, dict]
            The loss and the new state of the Q-network.
        """

        states, actions, rewards, terminals, next_states = batch
        q_key, q_target_key = jax.random.split(key)

        q_values, net_state = forward(q_network, params, state.net_state, q_key, states)
        q_values = jnp.take_along_axis(q_values, actions.astype(int), axis=-1)

        q_values_target, _ = forward(q_network, state.params_target, state.net_state_target, q_target_key, next_states)
        target = rewards + (1 - terminals) * discount * jnp.max(q_values_target, axis=-1, keepdims=True)

        target = jax.lax.stop_gradient(target)
        loss = optax.l2_loss(q_values, target).mean()

        return loss, net_state

    @staticmethod
    def update(
            state: DDQNState,
            key: PRNGKey,
            env_state: Array,
            action: Array,
            reward: Scalar,
            terminal: bool,
            step_fn: Callable,
            er: ExperienceReplay,
            experience_replay_steps: int,
            epsilon_decay: Scalar,
            epsilon_min: Scalar,
            tau: Scalar
    ) -> DDQNState:
        r"""
        Appends the transition to the experience replay buffer and performs ``experience_replay_steps`` steps.
        Each step consists of sampling a batch of transitions from the experience replay buffer, calculating the loss
        using the ``loss_fn`` function, performing a gradient step on the main Q-network, and soft updating the target
        Q-network. Soft update of the parameters is defined as :math:`\theta_{target} = \tau \theta + (1 - \tau) \theta_{target}`.
        The :math:`\epsilon`-greedy parameter is decayed by ``epsilon_decay``.

        Parameters
        ----------
        state : DDQNState
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
        er : ExperienceReplay
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
        DDQNState
            The updated state of the double Q-learning agent.
        """

        replay_buffer = er.append(state.replay_buffer, state.prev_env_state, action, reward, terminal, env_state)

        def for_loop_fn(_: int, carry: tuple) -> tuple:
            params, net_state, params_target, net_state_target, opt_state, key = carry
            batch_key, network_key, key = jax.random.split(key, 3)

            loss_params = (network_key, state, er.sample(replay_buffer, batch_key))
            params, net_state, opt_state, _ = step_fn(params, loss_params, opt_state)
            params_target, net_state_target = optax.incremental_update((params, net_state), (params_target, net_state_target), tau)

            return params, net_state, params_target, net_state_target, opt_state, key

        params, net_state, params_target, net_state_target, opt_state, _ = jax.lax.fori_loop(
            0, experience_replay_steps * er.is_ready(replay_buffer), for_loop_fn,
            (state.params, state.net_state, state.params_target, state.net_state_target, state.opt_state, key)
        )

        return DDQNState(
            params=params,
            net_state=net_state,
            params_target=params_target,
            net_state_target=net_state_target,
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=env_state,
            epsilon=jax.lax.max(state.epsilon * epsilon_decay, epsilon_min)
        )

    @staticmethod
    def sample(
            state: DDQNState,
            key: PRNGKey,
            env_state: Array,
            q_network: nn.Module,
            act_space_size: int
    ) -> int:
        r"""
        Samples random action with probability :math:`\epsilon` and the greedy action with probability
        :math:`1 - \epsilon` using the main Q-network. The greedy action is the action with the highest Q-value.

        Parameters
        ----------
        state : DDQNState
            The state of the double Q-learning agent.
        key : PRNGKey
            A PRNG key used as the random key.
        env_state : Array
            The current state of the environment.
        q_network : nn.Module
            The Q-network.
        act_space_size : int
            The size of the action space.

        Returns
        -------
        int
            Selected action.
        """

        network_key, action_key = jax.random.split(key)

        q, _ = forward(q_network, state.params, state.net_state, network_key, env_state)
        max_q = (q == q.max()).astype(float)
        probs = (1 - state.epsilon) * max_q / jnp.sum(max_q) + state.epsilon / q.shape[0]

        return jax.random.choice(action_key, act_space_size, p=probs.flatten())
