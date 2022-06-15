from typing import Tuple

import chex
import gym.spaces
import jax
import jax.numpy as jnp

from reinforced_lib.envs.env_state import EnvState
from reinforced_lib.envs.base_env import BaseEnv


@chex.dataclass
class Env_802_11_ax_DbgState(EnvState):
    """
    Container for the state of the 802.11ax debug environment.

    Fields
    ----------
    EnvState : _type_
        _description_
    """

    # TODO define the envirinment
