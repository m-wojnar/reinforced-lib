import json
from collections import defaultdict

from chex import Array, Scalar
from tensorboardX import SummaryWriter

from reinforced_lib.logs import BaseLogger, Source


class TensorboardLogger(BaseLogger):
    """
    Logger that saves values in TensorBoard [2]_ format. Offers a possibility to log to Comet [3]_.
    ``TensorboardLogger`` synchronizes the logged values in time. This means that if the same source
    is logged less often than other sources, the step will be increased accordingly to maintain the
    appropriate spacing between the values on the x-axis.

    Parameters
    ----------
    tb_log_dir : str, optional
        Path to the output directory. If None, the default directory is used.
    tb_comet_config : dict, optional
        Configuration for the Comet logger. If None, the logger is disabled.
    tb_sync_steps : bool, default=False
        Set to ``True`` if you want to synchronize the logged values in time.

    References
    ----------
    .. [2] TensorBoard. https://www.tensorflow.org/tensorboard
    .. [3] Comet. https://www.comet.ml
    """

    def __init__(
            self,
            tb_log_dir: str = None,
            tb_comet_config: dict[str, any] = None,
            tb_sync_steps: bool = False,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if tb_comet_config is None:
            tb_comet_config = {'disabled': True}

        self._sync_steps = tb_sync_steps
        self._current_values = set()
        self._step = 0

        self._writer = SummaryWriter(log_dir=tb_log_dir, comet_config=tb_comet_config)
        self._steps = defaultdict(int)

    def finish(self) -> None:
        """
        Closes the summary writer.
        """

        self._writer.close()

    def log_scalar(self, source: Source, value: Scalar, *_) -> None:
        """
        Adds a given scalar to the summary writer.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : float
            Scalar to log.
        """

        name = self.source_to_name(source)
        step = self._get_step(name)
        self._writer.add_scalar(name, value, step)

    def log_array(self, source: Source, value: Array, *_) -> None:
        """
        Adds a given array to the summary writer as a histogram.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : array_like
            Array to log.
        """

        name = self.source_to_name(source)
        step = self._get_step(name)
        self._writer.add_histogram(name, value, step)

    def log_dict(self, source: Source, value: dict, *_) -> None:
        """
        Logs a dictionary as a JSON string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : dict
            Dictionary to log.
        """

        self.log_other(source, value, None)

    def log_other(self, source: Source, value: any, *_) -> None:
        """
        Logs an object as a JSON string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : dict
            Dictionary to log.
        """

        name = self.source_to_name(source)
        step = self._get_step(name)
        self._writer.add_text(name, json.dumps(value), step)

    def _get_step(self, name: str) -> int:
        """
        Returns the current step for a given source.

        Parameters
        ----------
        name : str
            Name of the source.

        Returns
        -------
        int
            Current step for the given source.
        """

        if self._sync_steps:
            if name in self._current_values:
                self._step += 1
                self._current_values.clear()

            self._current_values.add(name)
            step = self._step
        else:
            step = self._steps[name] + 1

        self._steps[name] = step
        return step
