import json
from collections import defaultdict

from chex import Array, Scalar
from tensorboardX import SummaryWriter

from reinforced_lib.logs import BaseLogger, Source


class TensorboardLogger(BaseLogger):
    """
    Logger that saves values in TensorBoard [2]_ format. Offers a possibility to log to Comet [3]_.

    Parameters
    ----------
    tb_log_dir : str, optional
        Path to the output directory. If None, the default directory is used.
    tb_comet_config : dict, optional
        Configuration for the Comet logger. If None, the logger is disabled.

    References
    ----------
    .. [2] TensorBoard. https://www.tensorflow.org/tensorboard
    .. [3] Comet. https://www.comet.ml
    """

    def __init__(
            self,
            tb_log_dir: str = None,
            tb_comet_config: dict[str, any] = None,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if tb_comet_config is None:
            tb_comet_config = {'disabled': True}

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

        self._writer.add_scalar(self.source_to_name(source), value, self._steps[source])
        self._steps[source] += 1

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

        self._writer.add_histogram(self.source_to_name(source), value, self._steps[source])
        self._steps[source] += 1

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

        self._writer.add_text(self.source_to_name(source), json.dumps(value), self._steps[source])
        self._steps[source] += 1
