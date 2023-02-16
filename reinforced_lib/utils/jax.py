from typing import Callable, Tuple, Any

import jax
import optax


def gradient_step(
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        loss_params: Tuple,
        objective: Any,
        opt_state: optax.OptState
) -> Tuple[Any, optax.OptState]:
    """
    Performs a gradient step on the objective with respect to ``grad_loss_fn`` function.

    Parameters
    ----------
    optimizer : optax.GradientTransformation
        Optimizer to use for gradient step.
    loss_fn : Callable
        Function that returns the loss to be minimized.
    loss_params : Tuple
        Parameters to pass to ``loss_fn``.
    objective : Any
        Objective to be optimized.
    opt_state : optax.OptState
        Optimizer state.

    Returns
    -------
    out : tuple[Any, optax.OptState]
        Tuple containing the updated objective and optimizer state.
    """

    grads = jax.grad(loss_fn)(*loss_params)
    updates, opt_state = optimizer.update(grads, opt_state)
    objective = optax.apply_updates(objective, updates)

    return objective, opt_state
