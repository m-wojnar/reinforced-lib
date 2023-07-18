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
class ExpectedSarsaState(AgentState):
    """
    Container for the state of the deep expected SARSA agent.

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
    """

    params: hk.Params
    state: hk.State
    opt_state: optax.OptState

    replay_buffer: ReplayBuffer
    prev_env_state: Array


class ExpectedSarsa(BaseAgent):
    r"""
    Deep expected SARSA agent with temperature parameter :math:`\tau` and experience replay buffer. The agent uses
    a deep neural network to approximate the Q-value function. The Q-network is trained to minimize the Bellman
    error. This agent follows the on-policy learning paradigm and is suitable for environments with discrete action
    spaces.

    Parameters
    ----------
    q_network : hk.TransformedWithState
        Architecture of the Q-network.
    obs_space_shape : Shape
        Shape of the observation space.
    act_space_size : jnp.int32
        Size of the action space.
    optimizer : optax.GradientTransformation, optional
        Optimizer of the Q-network. If None, the Adam optimizer with learning rate 1e-3 is used.
    experience_replay_buffer_size : jnp.int32, default=10000
        Size of the experience replay buffer.
    experience_replay_batch_size : jnp.int32, default=64
        Batch size of the samples from the experience replay buffer.
    experience_replay_steps : jnp.int32, default=5
        Number of experience replay steps per update.
    discount : Scalar, default=0.99
        Discount factor. :math:`\gamma = 0.0` means no discount, :math:`\gamma = 1.0` means infinite discount. :math:`0 \leq \gamma \leq 1`
    tau : Scalar, default=1.0
        Temperature parameter. :math:`\tau = 0.0` means no exploration, :math:`\tau = \infty` means infinite exploration. :math:`\tau > 0`
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
            tau: Scalar = 1.0
    ) -> None:

        assert experience_replay_buffer_size > experience_replay_batch_size > 0
        assert 0.0 <= discount <= 1.0
        assert tau > 0.0

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
            experience_replay=er
        ))
        self.update = jax.jit(partial(
            self.update,
            q_network=q_network,
            step_fn=partial(
                gradient_step,
                optimizer=optimizer,
                loss_fn=partial(self.loss_fn, q_network=q_network, discount=discount, tau=tau)
            ),
            experience_replay=er,
            experience_replay_steps=experience_replay_steps
        ))
        self.sample = jax.jit(partial(
            self.sample,
            q_network=q_network,
            act_space_size=act_space_size,
            tau=tau
        ))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'obs_space_shape': gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), jnp.int32)),
            'act_space_size': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'experience_replay_buffer_size': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'experience_replay_batch_size': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32),
            'discount': gym.spaces.Box(0.0, 1.0, (1,)),
            'tau': gym.spaces.Box(0.0, jnp.inf, (1,))
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
            experience_replay: ExperienceReplay
    ) -> ExpectedSarsaState:
        r"""
        Initializes the Q-network, optimizer and experience replay buffer with given parameters.
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

        Returns
        -------
        ExpectedSarsaState
            Initial state of the deep expected SARSA agent.
        """

        x_dummy = jnp.empty(obs_space_shape)
        params, state = q_network.init(key, x_dummy)

        opt_state = optimizer.init(params)
        replay_buffer = experience_replay.init()

        return ExpectedSarsaState(
            params=params,
            state=state,
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=jnp.zeros(obs_space_shape)
        )

    @staticmethod
    def loss_fn(
            params: hk.Params,
            key: PRNGKey,
            net_state: hk.State,
            params_target: hk.Params,
            net_state_target: hk.State,
            batch: tuple,
            non_zero_loss: jnp.bool_,
            q_network: hk.TransformedWithState,
            discount: Scalar,
            tau: Scalar
    ) -> tuple[Scalar, hk.State]:
        r"""
        Loss is the mean squared Bellman error :math:`\mathcal{L}(\theta) = \mathbb{E}_{s, a, r, s'} \left[ \left( r +
        \gamma \sum_{a'} \pi(a'|s') Q(s', a') - Q(s, a) \right)^2 \right]` where :math:`s` is the current state,
        :math:`a` is the current action, :math:`r` is the reward, :math:`s'` is the next state, :math:`\gamma` is
        the discount factor, :math:`Q(s, a)` is the Q-value of the state-action pair. Loss can be calculated on a batch
        of transitions.

        Parameters
        ----------
        params : hk.Params
            The parameters of the Q-network.
        key : PRNGKey
            A PRNG key used as the random key.
        net_state : hk.State
            The state of the Q-network.
        params_target : hk.Params
            The parameters of the target Q-network.
        net_state_target : hk.State
            The state of the target Q-network.
        batch : tuple
            A batch of transitions from the experience replay buffer.
        non_zero_loss : bool
            Flag used to avoid updating the Q-network when the experience replay buffer is not full.
        q_network : hk.TransformedWithState
            The Q-network.
        discount : Scalar
            The discount factor.
        tau : Scalar
            The temperature parameter.

        Returns
        -------
        Tuple[Scalar, hk.State]
            The loss and the new state of the Q-network.
        """

        states, actions, rewards, terminals, next_states = batch
        q_key, q_target_key = jax.random.split(key)

        q_values, state = q_network.apply(params, net_state, q_key, states)
        q_values = jnp.take_along_axis(q_values, actions.astype(jnp.int32), axis=-1)

        q_values_target, _ = q_network.apply(params_target, net_state_target, q_target_key, next_states)
        probs_target = jax.nn.softmax(q_values_target / tau)
        target = rewards + (1 - terminals) * discount * jnp.sum(probs_target * q_values_target, axis=-1, keepdims=True)

        target = jax.lax.stop_gradient(target)
        loss = optax.l2_loss(q_values, target).mean()

        return loss * non_zero_loss, state

    @staticmethod
    def update(
            state: ExpectedSarsaState,
            key: PRNGKey,
            env_state: Array,
            action: Array,
            reward: Scalar,
            terminal: jnp.bool_,
            q_network: hk.TransformedWithState,
            step_fn: Callable,
            experience_replay: ExperienceReplay,
            experience_replay_steps: jnp.int32
    ) -> ExpectedSarsaState:
        r"""
        Appends the transition to the experience replay buffer and performs ``experience_replay_steps`` steps.
        Each step consists of sampling a batch of transitions from the experience replay buffer, calculating the loss
        using the ``loss_fn`` function and performing a gradient step on the Q-network.

        Parameters
        ----------
        state : ExpectedSarsaState
            The current state of the deep expected SARSA agent.
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
        q_network : hk.TransformedWithState
            The Q-network.
        step_fn : Callable
            The function that performs a single gradient step on the Q-network.
        experience_replay : ExperienceReplay
            The experience replay buffer.
        experience_replay_steps : int
            The number of experience replay steps.

        Returns
        -------
        ExpectedSarsaState
            The updated state of the deep expected SARSA agent.
        """

        replay_buffer = experience_replay.append(
            state.replay_buffer, state.prev_env_state,
            action, reward, terminal, env_state
        )

        params, net_state, opt_state = state.params, state.state, state.opt_state
        params_target, net_state_target = deepcopy(params), deepcopy(net_state)
        
        non_zero_loss = experience_replay.is_ready(replay_buffer)

        for _ in range(experience_replay_steps):
            batch_key, network_key, key = jax.random.split(key, 3)
            batch = experience_replay.sample(replay_buffer, batch_key)

            loss_params = (network_key, net_state, params_target, net_state_target, batch, non_zero_loss)
            params, net_state, opt_state, _ = step_fn(params, loss_params, opt_state)

        return ExpectedSarsaState(
            params=params,
            state=net_state,
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=env_state
        )

    @staticmethod
    def sample(
            state: ExpectedSarsaState,
            key: PRNGKey,
            env_state: Array,
            q_network: hk.TransformedWithState,
            act_space_size: jnp.int32,
            tau: Scalar
    ) -> jnp.int32:
        r"""
        Selects an action using the softmax policy with the temperature parameter :math:`\tau`:

        .. math::
            \pi(a|s) = \frac{e^{Q(s, a) / \tau}}{\sum_{a'} e^{Q(s, a') / \tau}}
        
        Parameters
        ----------
        state : ExpectedSarsaState
            The state of the deep expected SARSA agent.
        key : PRNGKey
            A PRNG key used as the random key.
        env_state : Array
            The current state of the environment.
        q_network : hk.TransformedWithState
            The Q-network.
        act_space_size : jnp.int32
            The size of the action space.
        tau : Scalar
            The temperature parameter.

        Returns
        -------
        int
            Selected action.
        """

        network_key, categorical_key = jax.random.split(key)

        logits = q_network.apply(state.params, state.state, network_key, env_state)[0]
        return jax.random.categorical(categorical_key, logits / tau)
