from typing import Any, Callable

import jax
import optax
from chex import Array, Scalar
from flax import linen as nn


def gradient_step(
        objective: any,
        loss_params: tuple,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable
) -> tuple[any, any, optax.OptState, Scalar]:
    r"""
    Performs a gradient step on the objective with respect to ``grad_loss_fn`` function.
    ``grad_loss_fn`` should return tuple of ``(loss, aux)`` where loss is the value to be minimized
    and aux is auxiliary value to be returned (can be ``None``).

    Parameters
    ----------
    objective : any
        Objective to be optimized.
    loss_params : tuple
        Parameters to pass to ``loss_fn``.
    opt_state : optax.OptState
        Optimizer state.
    optimizer : optax.GradientTransformation
        Optimizer to use for gradient step.
    loss_fn : Callable
        Function that returns the loss to be minimized. Can return additional values as well.

    Returns
    -------
    out : tuple[any, any, optax.OptState, Scalar]
        Tuple containing the updated objective and optimizer state, as well as the loss value.
    """

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(objective, *loss_params)
    updates, opt_state = optimizer.update(grads, opt_state)
    objective = optax.apply_updates(objective, updates)

    return objective, aux, opt_state, loss


def init(model: nn.Module, key: jax.random.PRNGKey, *x: Any) -> tuple[dict, dict]:
    r"""
    Initializes the ``flax`` model.

    Parameters
    ----------
    model : nn.Module
        Model to be initialized.
    key : PRNGKey
        A PRNG key used as the random key.
    x : any
        Input to the model.

    Returns
    -------
    tuple[dict, dict]
        Tuple containing the parameters and the state of the model.
    """

    params_key, rlib_key, dropout_key = jax.random.split(key, 3)

    variables = model.init({'params': params_key, 'rlib': rlib_key, 'dropout': dropout_key}, *x)
    params = variables.pop('params')
    state = variables

    return params, state


def forward(model: nn.Module, params: dict, state: dict, key: jax.random.PRNGKey, *x: Any) -> tuple[Array, dict]:
    r"""
    Forward pass through the ``flax`` model. **Note**: by default, the model is provided with two random key
    streams: one for the dropout layers and one for the user. This is done to ensure that the dropout is always
    initialized with the same random key, and that the user can use the custom key for any other purpose.
    The custom key is available in the model by calling ``self.make_rng('rlib')``.

    Parameters
    ----------
    model : nn.Module
        Model to be used for forward pass.
    params : dict
        Parameters of the model.
    state : dict
        State of the network.
    key : PRNGKey
        A PRNG key used as the random key.
    x : any
        Input to the model.

    Returns
    -------
    tuple[Array, dict]
        Tuple containing the output of the model and the updated state.
    """

    rlib_key, dropout_key = jax.random.split(key)
    return model.apply({'params': params, **state}, *x, rngs={'rlib': rlib_key, 'dropout': dropout_key}, mutable=list(state.keys()))
