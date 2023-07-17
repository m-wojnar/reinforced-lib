from typing import Callable

import jax
import optax
from chex import Scalar


def gradient_step(
        objective: any,
        loss_params: tuple,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable
) -> tuple[any, any, optax.OptState, Scalar]:
    """
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
