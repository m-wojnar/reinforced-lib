from chex import Array

import gymnasium as gym
import jax.numpy as jnp

from reinforced_lib.exts import BaseExt, observation, parameter


class IEEE_802_11_CCOD(BaseExt):
    """
    The IEEE 802.11 extension for the CCOD algorithm [1]_. Provides the preprocessed history
    of the transmission failure probability and default parameters for DRL agents.

    References
    ----------
    .. [1] W. Wydmański and S. Szott, "Contention Window Optimization in IEEE 802.11ax
       Networks with Deep Reinforcement Learning," 2021 IEEE Wireless Communications and
       Networking Conference (WCNC), 2021. https://doi.org/10.1109/WCNC49053.2021.9417575
    """

    max_history_length = 512

    def __init__(self, history_length: int) -> None:
        super().__init__()
        self.history_length = history_length

    observation_space = gym.spaces.Dict({
        'history': gym.spaces.Box(0, 1, (max_history_length,), float),
        'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float),
    })

    def preprocess(self, history: list) -> Array:
        """
        Preprocess the history according to the CCOD algorithm. The history is
        split into windows of equal length and the mean and standard deviation
        of each window is calculated. Window has a length of half the history
        and stride of quarter the history length. The resulting array has the
        shape (4, 2), where the first dimension corresponds to the window number
        and the second dimension corresponds to the mean and standard deviation
        respectively.

        Parameters
        ----------
        history : Array
            History of the transmission failure probability.

        Returns
        -------
        Array
            Preprocessed history.
        """

        history = jnp.array(history[:self.history_length])
        window = self.history_length // 2
        res = jnp.empty((4, 2))

        for i, pos in enumerate(range(0, self.history_length, window // 2)):
            res = res.at[i, 0].set(jnp.mean(history[pos:pos + window]))
            res = res.at[i, 1].set(jnp.std(history[pos:pos + window]))

        return jnp.clip(res, 0, 1)

    @observation(observation_type=gym.spaces.Box(-jnp.inf, jnp.inf, (4, 2), float))
    def env_state(self, history: list, *args, **kwargs) -> Array:
        return self.preprocess(history)

    @observation(observation_type=gym.spaces.MultiBinary(1))
    def terminal(self, *args, **kwargs) -> bool:
        return False

    @parameter(parameter_type=gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float))
    def min_reward(self) -> float:
        return 0.

    @parameter(parameter_type=gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float))
    def max_reward(self) -> float:
        return 1.

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(0, jnp.inf, (1,), int)))
    def obs_space_shape(self) -> tuple:
        return 4, 2

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, jnp.inf, (1,), int)))
    def act_space_shape(self) -> tuple:
        return tuple((1,))

    @parameter(parameter_type=gym.spaces.Box(1, jnp.inf, (1,), int))
    def act_space_size(self) -> int:
        return 7

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)))
    def min_action(self) -> tuple:
        return 0

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(-jnp.inf, jnp.inf, (1,), float)))
    def max_action(self) -> tuple:
        return 6
