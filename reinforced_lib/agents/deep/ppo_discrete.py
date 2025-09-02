from functools import partial
from typing import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from chex import dataclass, Array, PRNGKey, Scalar, Shape
from flax import linen as nn

from reinforced_lib.agents import BaseAgent, AgentState
from reinforced_lib.utils.rollout_buffer import rollout_buffer, RolloutBuffer, RolloutMemory
from reinforced_lib.utils.jax_utils import forward, gradient_step, init


@dataclass
class PPOState(AgentState):
    r"""
    Container for the state of the PPO agent.

    Attributes
    ----------
    params : dict
        Parameters of the agent network.
    net_state : dict
        State of the agent network.
    opt_state : optax.OptState
        Optimizer state.
    rollout_memory : RolloutMemory
        Rollout buffer storing the trajectories.
    prev_env_states : Array
        Previous environment state.
    counter : int
        Number of the current step during the rollout.
    """

    params: dict
    net_state: dict
    opt_state: optax.OptState

    rollout_memory: RolloutMemory
    prev_env_states: Array
    counter: int


class PPODiscrete(BaseAgent):
    r"""
    Proximal Policy Optimization (PPO) agent [5]_. This implementation uses the clipped surrogate objective.
    The policy and value functions should be represented by a single Flax module with two outputs: the action logits
    and the state value. The network should be able to process a batch of observations. The actions are sampled
    from a categorical distribution, while the value function is used to compute the advantages using Generalized
    Advantage Estimation (GAE) [6]_. The agent is trained using mini-batch gradient descent. This agent follows
    the on-policy learning paradigm and is suitable for environments with discrete action spaces.

    Parameters
    ----------
    network : nn.Module
        Architecture of the PPO agent network.
    obs_space_shape : Shape
        Shape of the observation space.
    act_space_size : int
        Size of the action space.
    optimizer : optax.GradientTransformation, optional
        Optimizer of the network. If None, the Adam optimizer with learning rate 3e-4 and :math:`\epsilon` = 1e-5 is used.
    discount : Scalar, default=0.99
        Discount factor. :math:`\gamma = 0.0` means no discount, :math:`\gamma = 1.0` means infinite discount. :math:`0 \leq \gamma \leq 1`
    lambda_gae : Scalar, default=0.9
        GAE parameter. :math:`\lambda = 0.0` means no GAE, :math:`\lambda = 1.0` means pure Monte Carlo advantage. :math:`0 \leq \lambda \leq 1`
    normalize_advantage : bool, default=True
        If True, the advantages are normalized to have mean 0 and standard deviation 1.
    clip_coef : Scalar, default=0.2
        Clipping coefficient for the surrogate objective, :math:`\epsilon` in [5]_.
    clip_value : bool, default=True
        If True, the loss for the value function is clipped.
    clip_grad : Scalar, default=0.5
        If not None, the gradients are clipped to have a maximum norm of `clip_grad`.
    entropy_coef : Scalar, default=0.01
        Coefficient for the entropy bonus.
    value_coef : Scalar, default=0.5
        Coefficient for the value function loss.
    rollout_length : int, default=512
        Length of the rollout buffer.
    num_envs : int, default=1
        Number of parallel environments.
    batch_size : int, default=128
        Size of the batch to be sampled from the rollout buffer.
    num_epochs : int, default=4
        Number of update epochs to perform on the rollout buffer.

    References
    ----------
    .. [5] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms.
    .. [6] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control
           using generalized advantage estimation.
    """

    def __init__(
            self,
            network: nn.Module,
            obs_space_shape: Shape,
            act_space_size: int,
            optimizer: optax.GradientTransformation = None,
            discount: Scalar = 0.99,
            lambda_gae: Scalar = 0.9,
            normalize_advantage: bool = True,
            clip_coef: Scalar = 0.2,
            clip_value: bool = True,
            clip_grad: Scalar = 0.5,
            entropy_coef: Scalar = 0.01,
            value_coef: Scalar = 0.5,
            rollout_length: int = 512,
            num_envs: int = 1,
            batch_size: int = 128,
            num_epochs: int = 4
    ) -> None:

        assert 0.0 <= discount <= 1.0
        assert 0.0 <= lambda_gae <= 1.0
        assert 0.0 <= clip_coef
        assert 0.0 <= entropy_coef
        assert 0.0 <= value_coef
        assert (rollout_length * num_envs) % batch_size == 0, 'rollout_length * num_envs must be divisible by batch_size'

        if optimizer is None:
            optimizer = optax.adam(3e-4, eps=1e-5)

        if clip_grad is not None:
            optimizer = optax.chain(optax.clip_by_global_norm(clip_grad), optimizer)

        self.obs_space_shape = obs_space_shape if jnp.ndim(obs_space_shape) > 0 else (obs_space_shape,)
        self.act_space_size = act_space_size
        self.num_envs = num_envs

        rb = rollout_buffer(
            rollout_length,
            num_envs,
            batch_size,
            discount,
            lambda_gae,
            self.obs_space_shape,
            (1,)
        )

        self.init = jax.jit(partial(
            self.init,
            num_envs=num_envs,
            obs_space_shape=self.obs_space_shape,
            network=network,
            optimizer=optimizer,
            rb=rb
        ))
        self.update = jax.jit(partial(
            self.update,
            network=network,
            step_fn=partial(
                gradient_step,
                optimizer=optimizer,
                loss_fn=partial(
                    self.loss_fn, network=network, normalize_advantage=normalize_advantage, clip_coef=clip_coef,
                    clip_value=clip_value, entropy_coef=entropy_coef, value_coef=value_coef
                )
            ),
            rb=rb,
            num_envs=num_envs,
            rollout_length=rollout_length,
            batch_size=batch_size,
            num_epochs=num_epochs
        ))
        self.sample = jax.jit(partial(
            self.sample,
            network=network
        ))

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'obs_space_shape': gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), int)),
            'act_space_size': gym.spaces.Box(1, jnp.inf, (1,), int),
            'discount': gym.spaces.Box(0.0, 1.0, (1,), float),
            'lambda_gae': gym.spaces.Box(0.0, 1.0, (1,), float),
            'normalize_advantage': gym.spaces.MultiBinary(1),
            'clip_coef': gym.spaces.Box(0.0, jnp.inf, (1,), float),
            'clip_value': gym.spaces.MultiBinary(1),
            'clip_grad': gym.spaces.Box(0.0, jnp.inf, (1,), float),
            'entropy_coef': gym.spaces.Box(0.0, jnp.inf, (1,), float),
            'value_coef': gym.spaces.Box(0.0, jnp.inf, (1,), float),
            'rollout_length': gym.spaces.Box(1, jnp.inf, (1,), int),
            'num_envs': gym.spaces.Box(1, jnp.inf, (1,), int),
            'batch_size': gym.spaces.Box(1, jnp.inf, (1,), int),
            'num_epochs': gym.spaces.Box(1, jnp.inf, (1,), int)
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_states': gym.spaces.Box(-jnp.inf, jnp.inf, (self.num_envs,) + self.obs_space_shape, float),
            'actions': gym.spaces.MultiDiscrete((self.act_space_size,) * self.num_envs),
            'rewards': gym.spaces.Box(-jnp.inf, jnp.inf, (self.num_envs,), float),
            'terminals': gym.spaces.MultiBinary(self.num_envs)
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            'env_states': gym.spaces.Box(-jnp.inf, jnp.inf, (self.num_envs,) + self.obs_space_shape, float)
        })

    @property
    def action_space(self) -> gym.spaces.MultiDiscrete:
        return gym.spaces.MultiDiscrete((self.act_space_size,) * self.num_envs)

    @staticmethod
    def init(
            key: PRNGKey,
            num_envs: int,
            obs_space_shape: Shape,
            network: nn.Module,
            optimizer: optax.GradientTransformation,
            rb: RolloutBuffer
    ) -> PPOState:
        r"""
        Initializes the PPO network, optimizer and rollout buffer with given parameters.
        The first state of the environment is assumed to be a tensor of zeros.

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        num_envs : int
            The number of parallel environments.
        obs_space_shape : Shape
            The shape of the observation space.
        network : nn.Module
            The agent network.
        optimizer : optax.GradientTransformation
            The optimizer.
        rb : RolloutBuffer
            The rollout buffer functions.

        Returns
        -------
        PPOState
            Initial state of the PPO agent.
        """

        x_dummy = jnp.empty((num_envs,) + obs_space_shape)
        params, net_state = init(network, key, x_dummy)

        opt_state = optimizer.init(params)
        rollout_memory = rb.init()

        return PPOState(
            params=params,
            net_state=net_state,
            opt_state=opt_state,
            rollout_memory=rollout_memory,
            prev_env_states=jnp.zeros_like(x_dummy),
            counter=0
        )

    @staticmethod
    def log_prob(logits: Array, actions: Array) -> tuple[Array, Array]:
        log_probs = jax.nn.log_softmax(logits)
        return log_probs, jnp.take_along_axis(log_probs, actions.astype(int), axis=-1).squeeze(axis=-1)

    @staticmethod
    def entropy(log_probs: Array) -> Array:
        return -(log_probs * jnp.exp(log_probs)).sum(axis=-1)

    @staticmethod
    def loss_fn(
            params: dict,
            key: PRNGKey,
            net_state: dict,
            batch: tuple,
            network: nn.Module,
            normalize_advantage: bool,
            clip_coef: Scalar,
            clip_value: bool,
            entropy_coef: Scalar,
            value_coef: Scalar
    ) -> tuple[Scalar, dict]:
        r"""
        Loss is the clipped surrogate objective with value function loss and entropy regularization:

        .. math::
            \mathcal{L}(\theta) =
            \mathbb{E}_t \Big[
                -\min\big(
                    A_t r_t(\theta),
                    A_t \, \mathrm{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)
                \big)
                + c_v \, \mathcal{L}_v(\theta)
                - c_e \, \mathcal{H}[\pi_\theta](s_t)
            \Big]

        where

        - :math:`r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}` is the
          probability ratio between the new and old policies,
        - :math:`A_t` is the advantage estimate,
        - :math:`\epsilon` is the clipping coefficient,
        - :math:`\mathcal{L}_v(\theta) = \tfrac{1}{2}\,(V_\theta(s_t) - R_t)^2` is the value function loss, possibly
          clipped,
        - :math:`R_t` is the discounted return,
        - :math:`\mathcal{H}[\pi_\theta](s_t)` is the entropy of the policy at state :math:`s_t`,
        - :math:`c_v` and :math:`c_e` are coefficients for the value loss and entropy bonus, respectively.

        Loss is calculated on a batch of transitions sampled from the rollout buffer.

        Parameters
        ----------
        params : dict
            The parameters of the agent network.
        key : PRNGKey
            A PRNG key used as the random key.
        net_state : dict
            The state of the agent network.
        batch : tuple
            A batch of transitions from the rollout buffer.
        network : nn.Module
            The agent network.
        normalize_advantage : bool
            If True, the advantages are normalized to have mean 0 and standard deviation 1.
        clip_coef : Scalar
            Clipping coefficient for the surrogate objective, :math:`\epsilon` in [5]_
        clip_value : bool
            If True, the loss for the value function is clipped.
        entropy_coef : Scalar
            Coefficient for the entropy bonus.
        value_coef : Scalar
            Coefficient for the value function loss.

        Returns
        -------
        Tuple[Scalar, dict]
            The loss and the new state of the agent network.
        """

        states, actions, _, _, values, log_probs, returns, advantages = batch

        (logits, new_values), net_state = forward(network, params, net_state, key, states)
        new_log_probs, new_log_probs_act = PPODiscrete.log_prob(logits, actions)

        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = jnp.exp(new_log_probs_act - log_probs)
        pg_loss1 = advantages * ratio
        pg_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        pg_loss = -jnp.minimum(pg_loss1, pg_loss2).mean()

        if clip_value:
            v_clipped = values + jnp.clip(new_values - values, -clip_coef, clip_coef)
            v_loss_clipped = optax.squared_error(v_clipped, returns)
            v_loss_unclipped = optax.squared_error(new_values, returns)
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * optax.squared_error(new_values, returns).mean()

        entropy = PPODiscrete.entropy(new_log_probs).mean()
        loss = pg_loss - entropy_coef * entropy + value_coef * v_loss

        return loss, net_state

    @staticmethod
    def update(
            state: PPOState,
            key: PRNGKey,
            env_states: Array,
            actions: Array,
            rewards: Scalar,
            terminals: bool,
            network: nn.Module,
            step_fn: Callable,
            rb: RolloutBuffer,
            num_envs: int,
            rollout_length: int,
            batch_size: int,
            num_epochs: int
    ) -> PPOState:
        r"""
        Appends the transition to the on-policy rollout buffer. Once the rollout buffer reaches ``rollout_length``
        steps, computes advantages and returns using GAE, shuffles and flattens the buffer, and performs multiple
        gradient updates using mini-batches.

        Parameters
        ----------
        state : PPOState
            The current state of the PPO agent.
        key : PRNGKey
            A PRNG key used as the random key.
        env_states : Array
            The current states of the environments.
        actions : Array
            The actions taken by the agent.
        rewards : Scalar
            The rewards received by the agent.
        terminals : bool
            Whether the episodes have terminated.
        network : nn.Module
            The agent network.
        step_fn : Callable
            The function that performs a single gradient step on the agent network.
        rb : RolloutBuffer
            The rollout buffer functions.
        rollout_length : int
            The length of the rollout buffer.
        num_envs : int
            The number of parallel environments.
        batch_size : int
            The size of the batch sampled from the rollout buffer.
        num_epochs : int
            The number of gradient steps to perform.

        Returns
        -------
        PPOState
            The updated state of the PPO agent.
        """

        network_key, update_key = jax.random.split(key)

        (logits, values), net_state = forward(network, state.params, state.net_state, network_key, state.prev_env_states)
        state = state.replace(net_state=net_state)
        actions = actions[..., None]

        _, log_probs_act = PPODiscrete.log_prob(logits, actions)
        rollout_memory = rb.append(state.rollout_memory, state.prev_env_states, actions, rewards, terminals, values, log_probs_act)

        def do_update(state: PPOState, rollout_memory: RolloutMemory, env_states: Array, key: PRNGKey) -> PPOState:
            network_key, shuffle_key, update_key = jax.random.split(key, 3)

            (_, last_values), net_state = forward(network, state.params, state.net_state, network_key, env_states)
            state = state.replace(net_state=net_state)

            rollout_memory = rb.compute_gae(rollout_memory, last_values)
            rollout_memory = rb.flatten_shuffle(rollout_memory, shuffle_key)
            state = state.replace(rollout_memory=rollout_memory)
            num_updates = (rollout_length * num_envs) // batch_size

            def body_fn(i: int, carry: tuple) -> tuple:
                key, state = carry
                key, step_key, shuffle_key = jax.random.split(key, 3)
                
                rollout_memory = jax.lax.cond(
                    i % num_updates == 0,
                    lambda rollout_memory: rb.flatten_shuffle(rollout_memory, shuffle_key),
                    lambda rollout_memory: rollout_memory,
                    state.rollout_memory
                )

                loss_params = (step_key, state.net_state, rb.get_batch(rollout_memory, i % num_updates))
                params, net_state, opt_state, _ = step_fn(state.params, loss_params, state.opt_state)

                return key, state.replace(
                    params=params,
                    net_state=net_state,
                    opt_state=opt_state,
                    rollout_memory=rollout_memory
                )

            _, state = jax.lax.fori_loop(0, num_epochs * num_updates, body_fn, (update_key, state))
            return state.replace(
                rollout_memory=rb.init(),
                prev_env_states=env_states,
                counter=0
            )

        def carry_on(state: PPOState, rollout_memory: RolloutMemory, env_states: Array, _) -> PPOState:
            return state.replace(
                rollout_memory=rollout_memory,
                prev_env_states=env_states,
                counter=state.counter + 1
            )

        return jax.lax.cond(
            state.counter + 1 == rollout_length, do_update, carry_on,
            state, rollout_memory, env_states, update_key
        )

    @staticmethod
    def sample(
            state: PPOState,
            key: PRNGKey,
            env_states: Array,
            network: nn.Module,
    ) -> Array:
        r"""
        Samples actions from the categorical distribution defined by the logits computed by the agent network.
        
        Parameters
        ----------
        state : PPOState
            The state of the PPO agent.
        key : PRNGKey
            A PRNG key used as the random key.
        env_states : Array
            The current state of the environment.
        network : nn.Module
            The agent network.

        Returns
        -------
        Array
            Selected actions.
        """

        network_key, action_key = jax.random.split(key)
        (logits, _), _ = forward(network, state.params, state.net_state, network_key, env_states)
        return jax.random.categorical(action_key, logits)
