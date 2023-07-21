import os.path
from collections import defaultdict

import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Array, Scalar, Numeric

from reinforced_lib.logs import BaseLogger, Source
from reinforced_lib.utils import timestamp


class PlotsLogger(BaseLogger):
    r"""
    Logger that presents and saves values as matplotlib plots. Offers smoothing of the curve, scatter plots, and
    multiple curves in a single chart (while logging arrays). ``PlotsLogger`` is able to synchronizes the logged
    values in time. This means that if the same source is logged less often than other sources, the step will be
    increased accordingly to maintain the appropriate spacing between the values on the x-axis.

    Parameters
    ----------
    plots_dir : str, default="~"
        Output directory for the plots.
    plots_ext : str, default="svg"
        Extension of the saved plots.
    plots_smoothing : float, default=0.6
        Weight of the exponential moving average (EMA/EWMA) [1]_ used for smoothing. :math:`\alpha \in [0, 1)`.
    plots_scatter : bool, default=False
        Set to ``True`` if you want to generate a scatter plot instead of a line plot.
        ``plots_smoothing`` parameter does not apply to the scatter plots.
    plots_sync_steps : bool, default=False
        Set to ``True`` if you want to synchronize the logged values in time.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    """

    def __init__(
            self,
            plots_dir: str = None,
            plots_ext: str = 'pdf',
            plots_smoothing: Scalar = 0.6,
            plots_scatter: bool = False,
            plots_sync_steps: bool = False,
            **kwargs
    ) -> None:
        assert 1 > plots_smoothing >= 0

        super().__init__(**kwargs)

        self._dir = plots_dir if plots_dir else os.path.expanduser("~")
        self._ext = plots_ext
        self._smoothing = plots_smoothing
        self._scatter = plots_scatter
        self._sync_steps = plots_sync_steps

        self._current_values = set()
        self._step = 0

        self._values = defaultdict(list)
        self._steps = defaultdict(list)

    def finish(self) -> None:
        """
        Shows the generated plots and saves them to the output directory with the specified extension
        (the names of the files follow the pattern ``"rlib-plot-[source]-[date]-[time].[ext]"``).
        """

        def exponential_moving_average(values: list, weight: Scalar) -> list:
            smoothed = [values[0]]

            for value in values[1:]:
                smoothed.append((1 - weight) * value + weight * smoothed[-1])

            return smoothed

        def lineplot(values: list, steps: list, alpha: Scalar = 1.0, label: bool = False) -> None:
            values = jnp.array(values)
            values = jnp.squeeze(values)

            if values.ndim == 1:
                plt.plot(steps, values, alpha=alpha, c='C0')
            elif values.ndim == 2:
                for i, val in enumerate(jnp.array(values).T):
                    plt.plot(steps, val, alpha=alpha, c=f'C{i % 10}', label=i if label else '')
                plt.legend()
        
        def scatterplot(values: list, steps: list, label: bool = False) -> None:
            values = jnp.array(values).squeeze()

            if values.ndim == 1:
                plt.scatter(steps, values, c='C0', marker='.', s=4)
            elif values.ndim == 2:
                for i, val in enumerate(jnp.array(values).T):
                    plt.scatter(steps, val, c=f'C{i % 10}', label=i if label else '', marker='.', s=4)
                plt.legend()

        for name, values in self._values.items():
            filename = f'rlib-plot-{name}-{timestamp()}.{self._ext}'

            if self._scatter:
                scatterplot(values, self._steps[name], True)
            else:
                smoothed = exponential_moving_average(values, self._smoothing)
                lineplot(values, self._steps[name], alpha=0.3)
                lineplot(smoothed, self._steps[name], label=True)
            
            plt.title(name)
            plt.xlabel('step')
            plt.savefig(os.path.join(self._dir, filename), bbox_inches='tight', dpi=300)
            plt.show()

    def log_scalar(self, source: Source, value: Scalar, *_) -> None:
        """
        Adds a given scalar to the plot values.

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
        Adds a given array to the plot values.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : array_like
            Array to log.
        """

        self._log(source, value)

    def _log(self, source: Source, value: Numeric) -> None:
        """
        Adds a given scalar to the plot values.

        Parameters
        ----------
        source : Source
            Source of the logged value.
        value : Numeric
            Value to log.
        """

        name = self.source_to_name(source)

        if self._sync_steps:
            if name in self._current_values:
                self._step += 1
                self._current_values.clear()

            self._current_values.add(name)
            step = self._step
        else:
            step = self._steps[name][-1] + 1 if self._steps[name] else 0

        self._values[name].append(value)
        self._steps[name].append(step)
