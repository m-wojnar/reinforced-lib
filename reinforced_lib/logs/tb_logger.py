import json
from typing import Any, Dict

from chex import Array, Scalar
from tensorboardX import SummaryWriter

from reinforced_lib.logs import BaseLogger, Source


class TensorboardLogger(BaseLogger):
    """
    Logger that saves values in TensorBoard [4]_ format. Offers a possibility to log to Comet [5]_.

    Parameters
    ----------
    tb_log_dir : str, optional
        Path to the output directory. If None, the default directory is used.
    tb_comet_config : dict, optional
        Configuration for the Comet logger. If None, the logger is disabled.

    References
    ----------
    .. [4] TensorBoard. https://www.tensorflow.org/tensorboard
    .. [5] Comet. https://www.comet.ml
    """

    def __init__(
            self,
            tb_log_dir: str = None,
            tb_comet_config: Dict[str, Any] = None,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if tb_comet_config is None:
            tb_comet_config = {'disabled': True}

        if tb_log_dir is None:
            self._summary_writer = SummaryWriter(comet_config=tb_comet_config)
        else:
            self._summary_writer = SummaryWriter(log_dir=tb_log_dir, comet_config=tb_comet_config)

        self._scalar_step = 0
        self._histogram_step = 0
        self._text_step = 0

    def finish(self) -> None:
        """
        Closes the summary writer.
        """

        self._summary_writer.close()

    def log_scalar(self, source: Source, value: Scalar) -> None:
        """
        Adds a given scalar to the summary writer.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : float
            Scalar to log.
        """

        self._summary_writer.add_scalar(self.source_to_name(source), value, self._scalar_step)
        self._scalar_step += 1

    def log_array(self, source: Source, value: Array) -> None:
        """
        Adds a given array to the summary writer as a histogram.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : array_like
            Array to log.
        """

        self._summary_writer.add_histogram(self.source_to_name(source), value, self._histogram_step)
        self._histogram_step += 1

    def log_dict(self, source: Source, value: Dict) -> None:
        """
        Logs a dictionary as a JSON [2]_ string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : dict
            Dictionary to log.
        """

        self.log_other(source, value)

    def log_other(self, source: Source, value: Any) -> None:
        """
        Logs an object as a JSON [2]_ string.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : dict
            Dictionary to log.
        """

        self._summary_writer.add_text(self.source_to_name(source), json.dumps(value), self._text_step)
        self._text_step += 1
