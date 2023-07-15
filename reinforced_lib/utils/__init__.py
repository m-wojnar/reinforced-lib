from datetime import datetime
from typing import Any

import numpy as np


def is_scalar(x: Any) -> bool:
    """
    Checks whether the input is a scalar.

    Parameters
    ----------
    x : any
        Input to check.

    Returns
    -------
    bool
        ``True`` if the input is a scalar, ``False`` otherwise.
    """

    return np.isscalar(x) or (hasattr(x, 'ndim') and x.ndim == 0)


def is_array(x: Any) -> bool:
    """
    Checks whether the input is an array.

    Parameters
    ----------
    x : any
        Input to check.

    Returns
    -------
    bool
        ``True`` if the input is an array, ``False`` otherwise.
    """

    return isinstance(x, (list, tuple)) or (hasattr(x, 'ndim') and x.ndim == 1)


def is_tensor(x: Any) -> bool:
    """
    Checks whether the input is a tensor.

    Parameters
    ----------
    x : any
        Input to check.

    Returns
    -------
    bool
        ``True`` if the input is a tensor, ``False`` otherwise.
    """

    return hasattr(x, 'ndim') and x.ndim > 1


def is_dict(x: Any) -> bool:
    """
    Checks whether the input is a dictionary.

    Parameters
    ----------
    x : any
        Input to check.

    Returns
    -------
    bool
        ``True`` if the input is a dictionary, ``False`` otherwise.
    """

    return isinstance(x, dict)


def timestamp() -> str:
    """
    Returns the current timestamp.

    Returns
    -------
    str
        Current timestamp.
    """

    return datetime.now().strftime('%Y%m%d-%H%M%S')
