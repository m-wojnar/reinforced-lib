from chex import Array

import gymnasium as gym
import numpy as np

from reinforced_lib.exts import BaseExt, observation, parameter


class IEEE_802_11_CCOD(BaseExt):
    """
    The IEEE 802.11 extension for the CCOD algorithm [3]_. Provides the preprocessed history
    of the transmission failure probability and default parameters for DRL agents.

    References
    ----------
    .. [3] W. Wydmański and S. Szott, "Contention Window Optimization in IEEE 802.11ax
       Networks with Deep Reinforcement Learning," 2021 IEEE Wireless Communications and
       Networking Conference (WCNC), 2021. https://doi.org/10.1109/WCNC49053.2021.9417575
    """

    max_history_length = 512

    def __init__(self, history_length: int) -> None:
        super().__init__()
        self.history_length = history_length

    observation_space = gym.spaces.Dict({
        'history': gym.spaces.Box(0, 1, (max_history_length,), np.float32),
        'reward': gym.spaces.Box(-np.inf, np.inf, (1,))
    })

    def preprocess(self, history: Array) -> Array:
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
        history : array_like
            History of the transmission failure probability.

        Returns
        -------
        array_like
            Preprocessed history.
        """

        history = history[:self.history_length]
        window = self.history_length // 2
        res = np.empty((4, 2))

        for i, pos in enumerate(range(0, self.history_length, window // 2)):
            res[i, 0] = np.mean(history[pos:pos + window])
            res[i, 1] = np.std(history[pos:pos + window])

        return np.clip(res, 0, 1)

    @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (4, 2), np.float32))
    def env_state(self, history: Array, *args, **kwargs) -> np.ndarray:
        return self.preprocess(history)

    @observation(observation_type=gym.spaces.MultiBinary(1))
    def terminal(self, *args, **kwargs) -> bool:
        return False

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def min_reward(self) -> float:
        return 0.

    @parameter(parameter_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def max_reward(self) -> float:
        return 1.

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(0, np.inf, (1,), np.int32)))
    def obs_space_shape(self) -> tuple:
        return 4, 2

    @parameter(parameter_type=gym.spaces.Sequence(gym.spaces.Box(1, np.inf, (1,), np.int32)))
    def act_space_shape(self) -> tuple:
        return tuple((1,))

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def act_space_size(self) -> int:
        return 7
