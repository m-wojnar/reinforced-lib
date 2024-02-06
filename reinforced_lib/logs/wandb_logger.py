from collections import defaultdict

import wandb
from chex import Array, Scalar

from reinforced_lib.logs import BaseLogger, Source


class WeightsAndBiasesLogger(BaseLogger):
    r"""
    Logger that saves values to Weights & Biases [4]_ platform. ``WeightsAndBiasesLogger`` synchronizes
    the logged values in time. This means that if the same source is logged less often than other sources,
    the step will be increased accordingly to maintain the appropriate spacing between the values on the x-axis.

    **Note**: to use this logger, you need to log into W&B before running the script. The necessary steps are
    described in the official documentation [4]_.

    Parameters
    ----------
    wandb_sync_steps : bool, default=False
        Set to ``True`` if you want to synchronize the logged values in time.
    wandb_kwargs : dict, optional
        Additional keyword arguments passed to ``wandb.init`` function.

    References
    ----------
    .. [4] Weights & Biases. https://docs.wandb.ai/
    """

    def __init__(
            self,
            wandb_sync_steps: bool = False,
            wandb_kwargs: dict = None,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._sync_steps = wandb_sync_steps
        self._current_values = set()
        self._step = 0
        self._steps = defaultdict(int)

        wandb.init(**(wandb_kwargs or {}))
        wandb.define_metric('*', step_metric='global_step')

    def finish(self) -> None:
        """
        Finishes the W&B run.
        """

        wandb.finish()

    def log_scalar(self, source: Source, value: Scalar, *_) -> None:
        """
        Logs a scalar value to the W&B logger.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : float
            Scalar to log.
        """

        self._log(source, value)

    def log_array(self, source: Source, value: Array, *_) -> None:
        """
        Logs an array to the W&B logger.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : Array
            Array to log.
        """

        self._log(source, value)

    def log_dict(self, source: Source, value: dict, *_) -> None:
        """
        Logs a dictionary to the W&B logger.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : dict
            Dictionary to log.
        """

        self._log(source, value)

    def log_other(self, source: Source, value: any, *_) -> None:
        """
        Logs an object to the W&B logger.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : any
            Value of any type to log.
        """

        self._log(source, value)

    def _log(self, source: Source, value: any) -> None:
        """
        Adds a given value to the logger.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : Numeric
            Value to log.
        """

        name = self.source_to_name(source)
        step = self._get_step(name)
        wandb.log({'global_step': step, name: value})

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
