import gymnasium as gym
import jax.numpy as jnp
from chex import Array, Scalar

from reinforced_lib.exts import BaseExt, observation, parameter


class IEEE_802_11_ax_RA(BaseExt):
    """
    The IEEE 802.11ax [1]_ extension. Provides the data rates (in Mb/s) for consecutive modulation and coding
    schemes (MCS), a reward calculated as the approximate throughput, environment state as a vector, and
    default parameters for MAB and DRL agents. This extension is adapted to the IEEE 802.11ax network with the
    following settings: the guard interval is equal to 3200 ns, channel width is 20 MHz, 1 spatial stream is used.

    References
    ----------
    .. [1] "IEEE Standard for Information Technology--Telecommunications and Information Exchange between Systems
       Local and Metropolitan Area Networks--Specific Requirements Part 11: Wireless LAN Medium Access Control
       (MAC) and Physical Layer (PHY) Specifications Amendment 1: Enhancements for High-Efficiency WLAN,"
       in IEEE Std 802.11ax-2021 (Amendment to IEEE Std 802.11-2020) , vol., no., pp.1-767,
       19 May 2021, doi: 10.1109/IEEESTD.2021.9442429.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_time = 0.0

    observation_space = gym.spaces.Dict({
        'time': gym.spaces.Box(0.0, jnp.inf, (1,), float),
        'n_successful': gym.spaces.Box(0, jnp.inf, (1,), int),
        'n_failed': gym.spaces.Box(0, jnp.inf, (1,), int),
        'n_wifi': gym.spaces.Box(1, jnp.inf, (1,), int),
        'power': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float),
        'cw': gym.spaces.Discrete(32767)
    })

    _wifi_modes_rates = jnp.array([
        7.3, 14.6, 21.9, 29.3, 43.9, 58.5, 65.8, 73.1, 87.8, 97.5, 109.7, 121.9
    ], dtype=float)

    @observation(observation_type=gym.spaces.Box(0.0, jnp.inf, (len(_wifi_modes_rates),), float))
    def rates(self, *args, **kwargs) -> Array:
        return self._wifi_modes_rates

    @observation(observation_type=gym.spaces.Box(-jnp.inf, jnp.inf, (len(_wifi_modes_rates),), float))
    def context(self, *args, **kwargs) -> Array:
        return self.rates()

    @observation(observation_type=gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float))
    def reward(self, action: int, n_successful: int, n_failed: int, *args, **kwargs) -> Scalar:
        if n_successful + n_failed > 0:
            return self._wifi_modes_rates[action] * n_successful / (n_successful + n_failed)
        else:
            return 0.0

    @observation(observation_type=gym.spaces.Box(0.0, jnp.inf, (1,), float))
    def delta_time(self, time: Scalar, *args, **kwargs) -> Scalar:
        delta_time = time - self.last_time
        self.last_time = time
        return delta_time

    @observation(observation_type=gym.spaces.Box(-jnp.inf, jnp.inf, (6,), float))
    def env_state(
            self,
            time: Scalar,
            n_successful: int,
            n_failed: int,
            n_wifi: int,
            power: Scalar,
            cw: int,
            *args,
            **kwargs
    ) -> Array:
        return jnp.array([self.delta_time(time), n_successful, n_failed, n_wifi, power, cw], dtype=float)

    @observation(observation_type=gym.spaces.MultiBinary(1))
    def terminal(self, *args, **kwargs) -> bool:
        return False

    @parameter(parameter_type=gym.spaces.Box(1, jnp.inf, (1,), int))
    def n_mcs(self) -> int:
        return len(self._wifi_modes_rates)

    @parameter(parameter_type=gym.spaces.Box(1, jnp.inf, (1,), int))
    def n_arms(self) -> int:
        return self.n_mcs()

    @parameter(parameter_type=gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float))
    def default_power(self) -> Scalar:
        return 16.0206

    @parameter(parameter_type=gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float))
    def min_reward(self) -> Scalar:
        return 0

    @parameter(parameter_type=gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float))
    def max_reward(self) -> int:
        return self._wifi_modes_rates.max()

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), int)))
    def obs_space_shape(self) -> tuple:
        return tuple((6,))

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), int)))
    def act_space_shape(self) -> tuple:
        return tuple((1,))

    @parameter(parameter_type=gym.spaces.Box(1, jnp.inf, (1,), int))
    def act_space_size(self) -> int:
        return 12

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)))
    def min_action(self) -> tuple:
        return 0

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)))
    def max_action(self) -> tuple:
        return 11
