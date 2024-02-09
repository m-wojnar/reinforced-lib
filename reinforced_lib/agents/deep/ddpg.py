from copy import deepcopy
from functools import partial
from typing import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from chex import dataclass, Array, PRNGKey, Scalar, Shape, Numeric
from flax import linen as nn

from reinforced_lib.agents import BaseAgent, AgentState
from reinforced_lib.utils.experience_replay import experience_replay, ExperienceReplay, ReplayBuffer
from reinforced_lib.utils.jax_utils import forward, gradient_step, init


@dataclass
class DDPGState(AgentState):
    r"""
    Container for the state of the deep deterministic policy gradient agent.

    Attributes
    ----------
    q_params : dict
        Parameters of the Q-network.
    q_net_state : dict
        State of the Q-network.
    q_params_target : dict
        Parameters of the target Q-network.
    q_net_state_target : dict
        State of the target Q-network.
    q_opt_state : optax.OptState
        Optimizer state of the Q-network.
    a_params : dict
        Parameters of the policy network.
    a_net_state : dict
        State of the policy network.
    a_params_target : dict
        Parameters of the target policy network.
    a_net_state_target : dict
        State of the target policy network.
    a_opt_state : optax.OptState
        Optimizer state of the policy network.
    replay_buffer : ReplayBuffer
        Experience replay buffer.
    prev_env_state : Array
        Previous environment state.
    noise : Scalar
        Current noise level.
    """

    q_params: dict
    q_net_state: dict
    q_params_target: dict
    q_net_state_target: dict
    q_opt_state: optax.OptState

    a_params: dict
    a_net_state: dict
    a_params_target: dict
    a_net_state_target: dict
    a_opt_state: optax.OptState

    replay_buffer: ReplayBuffer
    prev_env_state: Array
    noise: Scalar


class DDPG(BaseAgent):
    r"""
    Deep deterministic policy gradient [3]_ [4]_ agent with white Gaussian noise exploration and experience replay
    buffer. The agent simultaneously learns a Q-function and a policy. The Q-function is updated using the Bellman
    equation. The policy is learned using the gradient of the Q-function with respect to the policy parameters
    to maximize the Q-value. The agent uses two Q-networks and two policy networks to stabilize the learning process
    and avoid overestimation. The target networks are updated with a soft update. This agent follows the off-policy
    learning paradigm and is suitable for environments with continuous action spaces.

    Parameters
    ----------
    q_network : nn.Module
        Architecture of the Q-networks.
        The input to the network should be two tensors of observations and actions respectively.
    a_network : nn.Module
        Architecture of the policy networks.
    obs_space_shape : Shape
        Shape of the observation space.
    act_space_shape : Shape
        Shape of the action space.
    min_action : Scalar or Array
        Minimum action value.
    max_action : Scalar or Array
        Maximum action value.
    q_optimizer : optax.GradientTransformation, optional
        Optimizer of the Q-networks. If None, the Adam optimizer with learning rate 1e-3 is used.
    a_optimizer : optax.GradientTransformation, optional
        Optimizer of the policy networks. If None, the Adam optimizer with learning rate 1e-3 is used.
    experience_replay_buffer_size : int, default=10000
        Size of the experience replay buffer.
    experience_replay_batch_size : int, default=64
        Batch size of the samples from the experience replay buffer.
    experience_replay_steps : int, default=5
        Number of experience replay steps per update.
    discount : Scalar, default=0.99
        Discount factor. :math:`\gamma = 0.0` means no discount, :math:`\gamma = 1.0` means infinite discount. :math:`0 \leq \gamma \leq 1`
    noise : Scalar, default=(max_action - min_action) / 2
        Initial Gaussian noise level. :math:`0 \leq \sigma`.
    noise_decay : Scalar, default=0.99
        Gaussian noise decay factor. :math:`\sigma_{t+1} = \sigma_{t} * \sigma_{decay}`. :math:`0 \leq \sigma_{decay} \leq 1`.
    noise_min : Scalar, default=0.01
        Minimum Gaussian noise level. :math:`0 \leq \sigma_{min} \leq \sigma`.
    tau : Scalar, default=0.01
        Soft update factor. :math:`\tau = 0.0` means no soft update, :math:`\tau = 1.0` means hard update. :math:`0 \leq \tau \leq 1`.

    References
    ----------
    .. [3] David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. 2014.
       Deterministic policy gradient algorithms. In Proceedings of the 31st International Conference on International
       Conference on Machine Learning - Volume 32 (ICML'14). JMLR.org, I–387–I–395.
    .. [4] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver,
       and Daan Wierstra. 2015. Continuous control with deep reinforcement learning. CoRR abs/1509.02971.
    """

    def __init__(
            self,
            q_network: nn.Module,
            a_network: nn.Module,
            obs_space_shape: Shape,
            act_space_shape: Shape,
            min_action: Numeric,
            max_action: Numeric,
            q_optimizer: optax.GradientTransformation = None,
            a_optimizer: optax.GradientTransformation = None,
            experience_replay_buffer_size: int = 10000,
            experience_replay_batch_size: int = 64,
            experience_replay_steps: int = 5,
            discount: Scalar = 0.99,
            noise: Scalar = None,
            noise_decay: Scalar = 0.99,
            noise_min: Scalar = 0.01,
            tau: Scalar = 0.01
    ) -> None:

        assert experience_replay_buffer_size > experience_replay_batch_size > 0
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= noise_decay <= 1.0
        assert 0.0 <= tau <= 1.0

        if noise is None:
            noise = (max_action - min_action) / 2

        assert 0.0 <= noise
        assert 0.0 <= noise_min <= noise

        if q_optimizer is None:
            q_optimizer = optax.adam(1e-3)
        if a_optimizer is None:
            a_optimizer = optax.adam(1e-3)

        self.obs_space_shape = obs_space_shape if jnp.ndim(obs_space_shape) > 0 else (obs_space_shape,)
        self.act_space_shape = act_space_shape if jnp.ndim(act_space_shape) > 0 else (act_space_shape,)

        er = experience_replay(
            experience_replay_buffer_size,
            experience_replay_batch_size,
            self.obs_space_shape,
            self.act_space_shape
        )

        self.init = jax.jit(partial(
            self.init,
            obs_space_shape=self.obs_space_shape,
            act_space_shape=self.act_space_shape,
            q_network=q_network,
            a_network=a_network,
            q_optimizer=q_optimizer,
            a_optimizer=a_optimizer,
            er=er,
            noise=noise
        ))
        self.update = jax.jit(partial(
            self.update,
            q_step_fn=partial(
                gradient_step,
                optimizer=q_optimizer,
                loss_fn=partial(self.q_loss_fn, q_network=q_network, a_network=a_network, discount=discount)
            ),
            a_step_fn=partial(
                gradient_step,
                optimizer=a_optimizer,
                loss_fn=partial(self.a_loss_fn, q_network=q_network, a_network=a_network)
            ),
            er=er,
            experience_replay_steps=experience_replay_steps,
            noise_decay=noise_decay,
            noise_min=noise_min,
            tau=tau
        ))
        self.sample = jax.jit(partial(
            self.sample,
            a_network=a_network,
            min_action=min_action,
            max_action=max_action
        ))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'obs_space_shape': gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), int)),
            'act_space_shape': gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), int)),
            'min_action': gym.spaces.Sequence(gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)),
            'max_action': gym.spaces.Sequence(gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)),
            'experience_replay_buffer_size': gym.spaces.Box(1, jnp.inf, (1,), int),
            'experience_replay_batch_size': gym.spaces.Box(1, jnp.inf, (1,), int),
            'discount': gym.spaces.Box(0.0, 1.0, (1,), float),
            'noise': gym.spaces.Box(0.0, jnp.inf, (1,), float),
            'noise_decay': gym.spaces.Box(0.0, 1.0, (1,), float),
            'noise_min': gym.spaces.Box(0.0, jnp.inf, (1,), float),
            'tau': gym.spaces.Box(0.0, 1.0, (1,), float)
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_state': gym.spaces.Box(-jnp.inf, jnp.inf, self.obs_space_shape, float),
            'action': gym.spaces.Box(-jnp.inf, jnp.inf, self.act_space_shape, float),
            'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float),
            'terminal': gym.spaces.MultiBinary(1)
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_state': gym.spaces.Box(-jnp.inf, jnp.inf, self.obs_space_shape, float)
        })

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-jnp.inf, jnp.inf, self.act_space_shape, float)

    @staticmethod
    def init(
            key: PRNGKey,
            obs_space_shape: Shape,
            act_space_shape: Shape,
            q_network: nn.Module,
            a_network: nn.Module,
            q_optimizer: optax.GradientTransformation,
            a_optimizer: optax.GradientTransformation,
            er: ExperienceReplay,
            noise: Scalar
    ) -> DDPGState:
        r"""
        Initializes the Q-networks and the policy networks, optimizers, and experience replay buffer.
        The first state of the environment is assumed to be a tensor of zeros.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        obs_space_shape : Shape
            The shape of the observation space.
        act_space_shape : Shape
            The shape of the action space.
        q_network : nn.Module
            The Q-network.
        a_network : nn.Module
            The policy network.
        q_optimizer : optax.GradientTransformation
            The Q-network optimizer.
        a_optimizer : optax.GradientTransformation
            The policy network optimizer.
        er : ExperienceReplay
            The experience replay buffer.
        noise : Scalar
            The initial noise value.

        Returns
        -------
        DDPGState
            Initial state of the deep deterministic policy gradient agent.
        """

        s_dummy = jnp.empty(obs_space_shape)
        a_dummy = jnp.empty(act_space_shape)

        key, q_key, a_key = jax.random.split(key, 3)

        q_params, q_net_state = init(q_network, q_key, s_dummy, a_dummy)
        a_params, a_net_state = init(a_network, a_key, s_dummy)

        q_opt_state = q_optimizer.init(q_params)
        a_opt_state = a_optimizer.init(a_params)
        replay_buffer = er.init()

        return DDPGState(
            q_params=q_params,
            q_net_state=q_net_state,
            q_params_target=deepcopy(q_params),
            q_net_state_target=deepcopy(q_net_state),
            q_opt_state=q_opt_state,
            a_params=a_params,
            a_net_state=a_net_state,
            a_params_target=deepcopy(a_params),
            a_net_state_target=deepcopy(a_net_state),
            a_opt_state=a_opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=jnp.zeros(obs_space_shape),
            noise=noise
        )

    @staticmethod
    def q_loss_fn(
            q_params: dict,
            key: PRNGKey,
            state: DDPGState,
            batch: tuple,
            q_network: nn.Module,
            a_network: nn.Module,
            discount: Scalar
    ) -> tuple[Scalar, dict]:
        r"""
        Loss is the mean squared Bellman error :math:`\mathcal{L}(\theta) = \mathbb{E}_{s, a, r, s'} \left[ \left( r
        + \gamma \max Q'(s', \pi'(s')) - Q(s, a) \right)^2 \right]` where :math:`s` is the current state, :math:`a`
        is the current action, :math:`r` is the reward, :math:`s'` is the next state, :math:`\gamma` is  the discount
        factor, :math:`Q(s, a)` is the Q-value of the main Q-network, :math:`Q'(s, a)` is the Q-value of the target
        Q-network, and :math:`\pi'(s)` is the action of the target policy network. The policy network parameters
        are considered as fixed. Loss can be calculated on a batch of transitions.

        Parameters
        ----------
        q_params : dict
            The parameters of the Q-network.
        key : PRNGKey
            A PRNG key used as the random key.
        state : DDPGState
            The state of the deep deterministic policy gradient agent.
        batch : tuple
            A batch of transitions from the experience replay buffer.
        q_network : nn.Module
            The Q-network.
        a_network : nn.Module
            The policy network.
        discount : Scalar
            The discount factor.

        Returns
        -------
        tuple[Scalar, dict]
            The loss and the new state of the Q-network.
        """

        states, actions, rewards, terminals, next_states = batch
        q_key, q_target_key, a_target_key = jax.random.split(key, 3)

        q_values, q_net_state = forward(q_network, q_params, state.q_net_state, q_key, states, actions)

        actions_target, _ = forward(a_network, state.a_params_target, state.a_net_state_target, a_target_key, next_states)
        q_values_target, _ = forward(q_network, state.q_params_target, state.q_net_state_target, q_target_key, next_states, actions_target)
        target = rewards + (1 - terminals) * discount * q_values_target

        target = jax.lax.stop_gradient(target)
        loss = optax.l2_loss(q_values, target).mean()

        return loss, q_net_state

    @staticmethod
    def a_loss_fn(
            a_params: dict,
            key: PRNGKey,
            state: DDPGState,
            batch: tuple,
            q_network: nn.Module,
            a_network: nn.Module
    ) -> tuple[Scalar, dict]:
        r"""
        The policy network is updated using the gradient of the Q-network to maximize the Q-value of the current state
        and action :math:`\max_{\theta} \mathbb{E}_{s, a} \left[ Q(s, \pi_{\theta}(s)) \right]`. Q-network parameters are
        considered as fixed. The policy network can be updated on a batch of transitions.

        Parameters
        ----------
        a_params : dict
            The parameters of the policy network.
        key : PRNGKey
            A PRNG key used as the random key.
        state : DDPGState
            The state of the deep deterministic policy gradient agent.
        batch : tuple
            A batch of transitions from the experience replay buffer.
        q_network : nn.Module
            The Q-network.
        a_network : nn.Module
            The policy network.

        Returns
        -------
        tuple[Scalar, dict]
            The loss and the new state of the policy network.
        """

        states, _, _, _, _ = batch
        a_key, q_key = jax.random.split(key)

        actions, a_net_state = forward(a_network, a_params, state.a_net_state, a_key, states)
        q_values, _ = forward(q_network, state.q_params, state.q_net_state, q_key, states, actions)
        loss = -jnp.mean(q_values)

        return loss, a_net_state

    @staticmethod
    def update(
            state: DDPGState,
            key: PRNGKey,
            env_state: Array,
            action: Array,
            reward: Scalar,
            terminal: bool,
            q_step_fn: Callable,
            a_step_fn: Callable,
            er: ExperienceReplay,
            experience_replay_steps: int,
            noise_decay: Scalar,
            noise_min: Scalar,
            tau: Scalar
    ) -> DDPGState:
        r"""
        Appends the transition to the experience replay buffer and performs ``experience_replay_steps`` steps.
        Each step consists of sampling a batch of transitions from the experience replay buffer, calculating the
        Q-network loss and the policy network loss using ``q_loss_fn`` and ``a_loss_fn`` respectively, performing
        a gradient step on both networks, and soft updating the target networks.  Soft update of the parameters
        is defined as :math:`\theta_{target} = \tau \theta + (1 - \tau) \theta_{target}`.The noise parameter is
        decayed by ``noise_decay``.

        Parameters
        ----------
        state : DDPGState
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
        q_step_fn : Callable
            The function that performs a single gradient step on the Q-network.
        a_step_fn : Callable
            The function that performs a single gradient step on the policy network.
        er : ExperienceReplay
            The experience replay buffer.
        experience_replay_steps : int
            The number of experience replay steps.
        noise_decay : Scalar
            The decay rate of the noise parameter.
        noise_min : Scalar
            The minimum value of the noise parameter.
        tau : Scalar
            The soft update parameter.

        Returns
        -------
        DDPGState
            The updated state of the deep deterministic policy gradient agent.
        """

        replay_buffer = er.append(state.replay_buffer, state.prev_env_state, action, reward, terminal, env_state)

        def for_loop_fn(_: int, carry: tuple) -> tuple:
            q_params, q_net_state, q_params_target, q_net_state_target, q_opt_state, \
            a_params, a_net_state, a_params_target, a_net_state_target, a_opt_state, key = carry

            batch_key, q_network_key, a_network_key, key = jax.random.split(key, 4)
            batch = er.sample(replay_buffer, batch_key)

            q_params, q_net_state, q_opt_state, _ = q_step_fn(q_params, (q_network_key, state, batch), q_opt_state)
            a_params, a_net_state, a_opt_state, _ = a_step_fn(a_params, (a_network_key, state, batch), a_opt_state)

            q_params_target, q_net_state_target = optax.incremental_update((q_params, q_net_state), (q_params_target, q_net_state_target), tau)
            a_params_target, a_net_state_target = optax.incremental_update((a_params, a_net_state), (a_params_target, a_net_state_target), tau)

            return q_params, q_net_state, q_params_target, q_net_state_target, q_opt_state, \
                   a_params, a_net_state, a_params_target, a_net_state_target, a_opt_state, key

        q_params, q_net_state, q_params_target, q_net_state_target, q_opt_state, \
        a_params, a_net_state, a_params_target, a_net_state_target, a_opt_state, _ = jax.lax.fori_loop(
            0, experience_replay_steps, for_loop_fn,
            (state.q_params, state.q_net_state, state.q_params_target, state.q_net_state_target, state.q_opt_state,
            state.a_params, state.a_net_state, state.a_params_target, state.a_net_state_target, state.a_opt_state, key)
        )

        return DDPGState(
            q_params=q_params,
            q_net_state=q_net_state,
            q_params_target=q_params_target,
            q_net_state_target=q_net_state_target,
            q_opt_state=q_opt_state,
            a_params=a_params,
            a_net_state=a_net_state,
            a_params_target=a_params_target,
            a_net_state_target=a_net_state_target,
            a_opt_state=a_opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=env_state,
            noise=jnp.maximum(state.noise * noise_decay, noise_min)
        )

    @staticmethod
    def sample(
            state: DDPGState,
            key: PRNGKey,
            env_state: Array,
            a_network: nn.Module,
            min_action: Scalar,
            max_action: Scalar
    ) -> Numeric:
        r"""
        Calculates deterministic action using the policy network. Then adds white Gaussian noise with standard
        deviation ``state.noise`` to the action and clips it to the range :math:`[min\_action, max\_action]`.

        Parameters
        ----------
        state : DDPGState
            The state of the double Q-learning agent.
        key : PRNGKey
            A PRNG key used as the random key.
        env_state : Array
            The current state of the environment.
        a_network : nn.Module
            The policy network.
        min_action : Scalar or Array
            The minimum value of the action.
        max_action : Scalar or Array
            The maximum value of the action.

        Returns
        -------
        Scalar or Array
            Selected action.
        """

        network_key, noise_key = jax.random.split(key)

        action, _ = forward(a_network, state.a_params, state.a_net_state, network_key, env_state)
        action += jax.random.normal(noise_key, action.shape) * state.noise

        return jnp.clip(action, min_action, max_action)
