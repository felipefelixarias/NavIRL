"""TensorBoard integration logger for NavIRL.

Provides a high-level interface for logging scalars, histograms, images,
text, hyperparameters, and custom metrics to TensorBoard. Handles the
optional ``tensorboard`` dependency gracefully via try/except.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter  # type: ignore[no-redef]

        _TB_AVAILABLE = True
    except ImportError:
        SummaryWriter = None  # type: ignore[assignment,misc]
        _TB_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_tensorboard_available() -> bool:
    """Check whether a TensorBoard backend is importable.

    Returns:
        ``True`` if ``torch.utils.tensorboard`` or ``tensorboardX`` can be
        imported.
    """
    return _TB_AVAILABLE


# ---------------------------------------------------------------------------
# Step tracker
# ---------------------------------------------------------------------------


class StepTracker:
    """Tracks global and per-tag step counters.

    Useful when different metric families advance at different rates
    (e.g. training steps vs. evaluation episodes).

    Attributes:
        global_step: The default step counter.
    """

    def __init__(self, initial_step: int = 0) -> None:
        self.global_step: int = initial_step
        self._tag_steps: dict[str, int] = {}

    def increment(self, n: int = 1) -> int:
        """Increment the global step counter.

        Args:
            n: Amount to increment by.

        Returns:
            The new global step value.
        """
        self.global_step += n
        return self.global_step

    def set(self, step: int) -> None:
        """Set the global step to an explicit value.

        Args:
            step: The step value to set.
        """
        self.global_step = step

    def get(self, tag: str | None = None) -> int:
        """Get the current step for a tag, or the global step.

        Args:
            tag: Optional tag name. If ``None``, returns ``global_step``.

        Returns:
            The step counter value.
        """
        if tag is None:
            return self.global_step
        return self._tag_steps.get(tag, self.global_step)

    def increment_tag(self, tag: str, n: int = 1) -> int:
        """Increment a tag-specific step counter.

        Args:
            tag: The tag whose counter to increment.
            n: Amount to increment by.

        Returns:
            The new step value for this tag.
        """
        current = self._tag_steps.get(tag, 0)
        self._tag_steps[tag] = current + n
        return self._tag_steps[tag]

    def set_tag(self, tag: str, step: int) -> None:
        """Set a tag-specific step to an explicit value.

        Args:
            tag: The tag name.
            step: The step value.
        """
        self._tag_steps[tag] = step


# ---------------------------------------------------------------------------
# Metric group helper
# ---------------------------------------------------------------------------


class MetricGroup:
    """Groups related metrics under a common prefix for organised logging.

    Args:
        logger: Parent ``TBLogger`` instance.
        prefix: String prefix prepended to all tags in this group.
    """

    def __init__(self, logger_ref: TBLogger, prefix: str) -> None:
        self._logger = logger_ref
        self._prefix = prefix.rstrip("/")

    def _tag(self, name: str) -> str:
        return f"{self._prefix}/{name}"

    def scalar(self, name: str, value: float, step: int | None = None) -> None:
        """Log a scalar metric under this group.

        Args:
            name: Metric name (will be prefixed).
            value: Scalar value.
            step: Step index. Uses the logger's global step if ``None``.
        """
        self._logger.add_scalar(self._tag(name), value, step=step)

    def scalars(self, main_tag: str, values: dict[str, float], step: int | None = None) -> None:
        """Log multiple scalars under one chart.

        Args:
            main_tag: Chart name (will be prefixed).
            values: Mapping of series name to value.
            step: Step index.
        """
        self._logger.add_scalars(self._tag(main_tag), values, step=step)

    def histogram(self, name: str, values: np.ndarray | Sequence[float], step: int | None = None, bins: str = "tensorflow") -> None:
        """Log a histogram under this group.

        Args:
            name: Metric name.
            values: Array of values.
            step: Step index.
            bins: Binning strategy.
        """
        self._logger.add_histogram(self._tag(name), values, step=step, bins=bins)


# ---------------------------------------------------------------------------
# Main logger
# ---------------------------------------------------------------------------


class TBLogger:
    """TensorBoard logger with step tracking, context managers, and grouping.

    Wraps a ``SummaryWriter`` and adds convenience methods for common NavIRL
    logging patterns such as training curves, evaluation metrics, trajectory
    images, reward distributions, loss curves, and learning rate schedules.

    Args:
        log_dir: Directory where TensorBoard event files are written.
        experiment_name: Optional experiment name used as a sub-directory.
        flush_secs: How often (seconds) to flush pending events.
        enabled: If ``False``, all logging calls become no-ops.  Useful for
            non-rank-0 processes in distributed training.
        initial_step: Starting value for the global step counter.

    Raises:
        ImportError: If *enabled* is ``True`` but TensorBoard is not installed.

    Example::

        with TBLogger("/tmp/tb_logs", experiment_name="run_01") as tb:
            for step in range(100):
                tb.add_scalar("train/loss", loss_val, step=step)
            tb.add_hparams({"lr": 1e-3}, {"final_loss": 0.01})
    """

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: str | None = None,
        flush_secs: int = 120,
        enabled: bool = True,
        initial_step: int = 0,
    ) -> None:
        self._enabled = enabled and _TB_AVAILABLE
        self._log_dir = Path(log_dir)
        self._experiment_name = experiment_name
        self._step_tracker = StepTracker(initial_step)
        self._groups: dict[str, MetricGroup] = {}
        self._closed = False
        self._wall_start = time.time()
        self._scalar_cache: dict[str, list[tuple[int, float]]] = {}

        if enabled and not _TB_AVAILABLE:
            raise ImportError(
                "TensorBoard is not installed. Install it with: "
                "pip install tensorboard  or  pip install tensorboardX"
            )

        if self._enabled:
            effective_dir = self._log_dir
            if experiment_name:
                effective_dir = self._log_dir / experiment_name
            effective_dir.mkdir(parents=True, exist_ok=True)
            self._writer: SummaryWriter | None = SummaryWriter(
                log_dir=str(effective_dir), flush_secs=flush_secs
            )
        else:
            self._writer = None

    # -- Properties -------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether this logger is active."""
        return self._enabled

    @property
    def global_step(self) -> int:
        """Current global step counter value."""
        return self._step_tracker.global_step

    @global_step.setter
    def global_step(self, value: int) -> None:
        self._step_tracker.set(value)

    @property
    def log_dir(self) -> Path:
        """Root log directory."""
        return self._log_dir

    @property
    def step_tracker(self) -> StepTracker:
        """The underlying step tracker."""
        return self._step_tracker

    # -- Context manager --------------------------------------------------------

    def __enter__(self) -> TBLogger:
        """Enter context manager.

        Returns:
            This logger instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and close the writer."""
        self.close()

    # -- Metric groups ----------------------------------------------------------

    def group(self, prefix: str) -> MetricGroup:
        """Get or create a metric group with the given prefix.

        Metric groups organise related scalars, histograms, etc. under a
        common TensorBoard tag prefix.

        Args:
            prefix: The prefix string (e.g. ``"train"`` or ``"eval"``).

        Returns:
            A ``MetricGroup`` instance bound to this logger.
        """
        if prefix not in self._groups:
            self._groups[prefix] = MetricGroup(self, prefix)
        return self._groups[prefix]

    # -- Scalars ----------------------------------------------------------------

    def _resolve_step(self, step: int | None) -> int:
        """Return the provided step or the current global step."""
        return step if step is not None else self._step_tracker.global_step

    def add_scalar(
        self,
        tag: str,
        value: float,
        step: int | None = None,
        wall_time: float | None = None,
    ) -> None:
        """Log a scalar value.

        Args:
            tag: TensorBoard tag (e.g. ``"train/loss"``).
            value: Scalar value.
            step: Step index.  Defaults to global step.
            wall_time: Optional wall-clock timestamp.
        """
        if not self._enabled or self._writer is None:
            return
        s = self._resolve_step(step)
        kwargs: dict[str, Any] = {"tag": tag, "scalar_value": float(value), "global_step": s}
        if wall_time is not None:
            kwargs["walltime"] = wall_time
        self._writer.add_scalar(**kwargs)

        # Cache for later retrieval
        self._scalar_cache.setdefault(tag, []).append((s, float(value)))

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        step: int | None = None,
        wall_time: float | None = None,
    ) -> None:
        """Log multiple scalars under one chart.

        Args:
            main_tag: Main chart tag.
            tag_scalar_dict: Mapping of series name to value.
            step: Step index.
            wall_time: Optional wall-clock timestamp.
        """
        if not self._enabled or self._writer is None:
            return
        s = self._resolve_step(step)
        kwargs: dict[str, Any] = {
            "main_tag": main_tag,
            "tag_scalar_dict": tag_scalar_dict,
            "global_step": s,
        }
        if wall_time is not None:
            kwargs["walltime"] = wall_time
        self._writer.add_scalars(**kwargs)

    # -- Training curves --------------------------------------------------------

    def log_training_step(
        self,
        step: int,
        loss: float,
        lr: float | None = None,
        grad_norm: float | None = None,
        extra: dict[str, float] | None = None,
    ) -> None:
        """Log metrics for a single training step.

        Convenience method that logs loss, learning rate, gradient norm, and
        any extra scalars under the ``train/`` prefix.

        Args:
            step: Training step index.
            loss: Loss value.
            lr: Optional learning rate.
            grad_norm: Optional gradient norm.
            extra: Optional mapping of additional scalar names to values.
        """
        self.add_scalar("train/loss", loss, step=step)
        if lr is not None:
            self.add_scalar("train/learning_rate", lr, step=step)
        if grad_norm is not None:
            self.add_scalar("train/grad_norm", grad_norm, step=step)
        if extra:
            for name, val in extra.items():
                self.add_scalar(f"train/{name}", val, step=step)

    def log_loss_components(
        self,
        step: int,
        components: dict[str, float],
        prefix: str = "loss",
    ) -> None:
        """Log individual loss components.

        Args:
            step: Training step.
            components: Mapping of component names to values.
            prefix: Tag prefix.
        """
        for name, val in components.items():
            self.add_scalar(f"{prefix}/{name}", val, step=step)
        total = sum(components.values())
        self.add_scalar(f"{prefix}/total", total, step=step)

    def log_learning_rate_schedule(
        self,
        step: int,
        lr: float,
        tag: str = "schedule/learning_rate",
    ) -> None:
        """Log the current learning rate.

        Args:
            step: Training step.
            lr: Current learning rate.
            tag: TensorBoard tag.
        """
        self.add_scalar(tag, lr, step=step)

    # -- Evaluation metrics -----------------------------------------------------

    def log_evaluation(
        self,
        step: int,
        metrics: dict[str, float],
        prefix: str = "eval",
    ) -> None:
        """Log evaluation metrics.

        Args:
            step: Evaluation step or episode number.
            metrics: Mapping of metric names to values.
            prefix: Tag prefix.
        """
        for name, val in metrics.items():
            self.add_scalar(f"{prefix}/{name}", val, step=step)

    def log_episode_summary(
        self,
        episode: int,
        reward: float,
        length: int,
        success: bool | None = None,
        extra: dict[str, float] | None = None,
    ) -> None:
        """Log per-episode summary metrics.

        Args:
            episode: Episode index.
            reward: Total episode reward.
            length: Episode length in steps.
            success: Whether the episode was successful.
            extra: Additional metrics.
        """
        self.add_scalar("episode/reward", reward, step=episode)
        self.add_scalar("episode/length", float(length), step=episode)
        if success is not None:
            self.add_scalar("episode/success", float(success), step=episode)
        if extra:
            for k, v in extra.items():
                self.add_scalar(f"episode/{k}", v, step=episode)

    # -- Histograms -------------------------------------------------------------

    def add_histogram(
        self,
        tag: str,
        values: np.ndarray | Sequence[float],
        step: int | None = None,
        bins: str = "tensorflow",
    ) -> None:
        """Log a histogram of values.

        Args:
            tag: TensorBoard tag.
            values: 1-D array or sequence of numerical values.
            step: Step index.
            bins: Binning strategy (default ``"tensorflow"``).
        """
        if not self._enabled or self._writer is None:
            return
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size == 0:
            return
        s = self._resolve_step(step)
        self._writer.add_histogram(tag, arr, global_step=s, bins=bins)

    def log_reward_distribution(
        self,
        step: int,
        rewards: np.ndarray | Sequence[float],
        tag: str = "reward/distribution",
    ) -> None:
        """Log the distribution of rewards as a histogram.

        Args:
            step: Step or episode index.
            rewards: Array of reward values.
            tag: TensorBoard tag.
        """
        self.add_histogram(tag, rewards, step=step)

    def log_weight_histograms(
        self,
        step: int,
        named_params: dict[str, np.ndarray],
        prefix: str = "weights",
    ) -> None:
        """Log histograms of model weight arrays.

        Args:
            step: Training step.
            named_params: Mapping of parameter names to numpy arrays.
            prefix: Tag prefix.
        """
        for name, arr in named_params.items():
            safe_name = name.replace(".", "/")
            self.add_histogram(f"{prefix}/{safe_name}", arr, step=step)

    def log_gradient_histograms(
        self,
        step: int,
        named_gradients: dict[str, np.ndarray],
        prefix: str = "gradients",
    ) -> None:
        """Log histograms of gradient arrays.

        Args:
            step: Training step.
            named_gradients: Mapping of parameter names to gradient arrays.
            prefix: Tag prefix.
        """
        for name, arr in named_gradients.items():
            safe_name = name.replace(".", "/")
            self.add_histogram(f"{prefix}/{safe_name}", arr, step=step)

    # -- Images -----------------------------------------------------------------

    def add_image(
        self,
        tag: str,
        image: np.ndarray,
        step: int | None = None,
        dataformats: str = "HWC",
    ) -> None:
        """Log an image.

        Args:
            tag: TensorBoard tag.
            image: Image array. Shape depends on *dataformats*.
            step: Step index.
            dataformats: Dimension ordering, e.g. ``"HWC"`` or ``"CHW"``.
        """
        if not self._enabled or self._writer is None:
            return
        s = self._resolve_step(step)
        self._writer.add_image(tag, image, global_step=s, dataformats=dataformats)

    def add_images(
        self,
        tag: str,
        images: np.ndarray,
        step: int | None = None,
        dataformats: str = "NHWC",
    ) -> None:
        """Log a batch of images.

        Args:
            tag: TensorBoard tag.
            images: Batch of images. Shape depends on *dataformats*.
            step: Step index.
            dataformats: Dimension ordering, e.g. ``"NHWC"`` or ``"NCHW"``.
        """
        if not self._enabled or self._writer is None:
            return
        s = self._resolve_step(step)
        self._writer.add_images(tag, images, global_step=s, dataformats=dataformats)

    def log_trajectory_image(
        self,
        step: int,
        positions: dict[int, np.ndarray],
        img_size: tuple[int, int] = (256, 256),
        world_bounds: tuple[float, float, float, float] | None = None,
        tag: str = "trajectory/visualization",
    ) -> None:
        """Render agent trajectories as an image and log to TensorBoard.

        Draws each agent's path on a blank canvas using simple numpy
        operations (no external rendering library required).

        Args:
            step: Step index.
            positions: Mapping of agent IDs to ``(N, 2)`` position arrays.
            img_size: ``(height, width)`` of the output image.
            world_bounds: ``(x_min, x_max, y_min, y_max)`` for coordinate
                mapping. If ``None``, bounds are inferred from data.
            tag: TensorBoard tag.
        """
        h, w = img_size
        img = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background

        if not positions:
            self.add_image(tag, img, step=step, dataformats="HWC")
            return

        # Determine world bounds
        if world_bounds is None:
            all_pts = np.concatenate(list(positions.values()), axis=0)
            x_min, y_min = all_pts.min(axis=0) - 1.0
            x_max, y_max = all_pts.max(axis=0) + 1.0
        else:
            x_min, x_max, y_min, y_max = world_bounds

        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        # Simple colour palette
        colours = [
            (255, 0, 0), (0, 0, 255), (0, 180, 0), (255, 165, 0),
            (128, 0, 128), (0, 128, 128), (255, 20, 147), (139, 69, 19),
        ]

        for idx, (_aid, pts) in enumerate(sorted(positions.items())):
            colour = colours[idx % len(colours)]
            for i in range(len(pts) - 1):
                px0 = int((pts[i, 0] - x_min) / x_range * (w - 1))
                py0 = int((1.0 - (pts[i, 1] - y_min) / y_range) * (h - 1))
                px1 = int((pts[i + 1, 0] - x_min) / x_range * (w - 1))
                py1 = int((1.0 - (pts[i + 1, 1] - y_min) / y_range) * (h - 1))

                # Bresenham-style line
                num_pts = max(abs(px1 - px0), abs(py1 - py0), 1)
                for t_i in range(num_pts + 1):
                    t = t_i / num_pts
                    px = int(px0 + t * (px1 - px0))
                    py = int(py0 + t * (py1 - py0))
                    if 0 <= px < w and 0 <= py < h:
                        # Draw a small dot
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                nx, ny = px + dx, py + dy
                                if 0 <= nx < w and 0 <= ny < h:
                                    img[ny, nx] = colour

        self.add_image(tag, img, step=step, dataformats="HWC")

    # -- Text -------------------------------------------------------------------

    def add_text(self, tag: str, text: str, step: int | None = None) -> None:
        """Log a text string.

        Args:
            tag: TensorBoard tag.
            text: Markdown-formatted text.
            step: Step index.
        """
        if not self._enabled or self._writer is None:
            return
        s = self._resolve_step(step)
        self._writer.add_text(tag, text, global_step=s)

    # -- Hyperparameters --------------------------------------------------------

    def add_hparams(
        self,
        hparam_dict: dict[str, Any],
        metric_dict: dict[str, float],
    ) -> None:
        """Log hyperparameters alongside final metrics.

        This creates a dedicated HParams entry in TensorBoard for comparing
        runs with different hyperparameter settings.

        Args:
            hparam_dict: Mapping of hyperparameter names to values.
                Values should be ``int``, ``float``, ``str``, or ``bool``.
            metric_dict: Mapping of metric names to final values.
        """
        if not self._enabled or self._writer is None:
            return
        # Sanitise hparam values
        clean_hp: dict[str, int | float | str | bool] = {}
        for k, v in hparam_dict.items():
            if isinstance(v, (int, float, str, bool)):
                clean_hp[k] = v
            else:
                clean_hp[k] = str(v)
        self._writer.add_hparams(clean_hp, metric_dict)

    def log_config(self, config: dict[str, Any], step: int = 0) -> None:
        """Log a configuration dictionary as a text block.

        Args:
            config: Configuration dictionary.
            step: Step index.
        """
        text = "```json\n" + json.dumps(config, indent=2, default=str) + "\n```"
        self.add_text("config", text, step=step)

    # -- Custom metrics ---------------------------------------------------------

    def log_custom_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        prefix: str = "custom",
    ) -> None:
        """Log a custom scalar metric.

        Args:
            name: Metric name.
            value: Metric value.
            step: Step index.
            prefix: Tag prefix.
        """
        self.add_scalar(f"{prefix}/{name}", value, step=step)

    def log_timing(
        self,
        name: str,
        duration_s: float,
        step: int | None = None,
    ) -> None:
        """Log a timing measurement.

        Args:
            name: Name of the timed operation.
            duration_s: Duration in seconds.
            step: Step index.
        """
        self.add_scalar(f"timing/{name}", duration_s, step=step)

    def log_throughput(
        self,
        name: str,
        count: int,
        duration_s: float,
        step: int | None = None,
    ) -> None:
        """Log a throughput measurement (items per second).

        Args:
            name: Name of the throughput metric.
            count: Number of items processed.
            duration_s: Duration in seconds.
            step: Step index.
        """
        rate = count / max(duration_s, 1e-9)
        self.add_scalar(f"throughput/{name}", rate, step=step)

    # -- Cached scalar retrieval ------------------------------------------------

    def get_scalar_history(self, tag: str) -> list[tuple[int, float]]:
        """Return cached scalar values for a tag.

        Args:
            tag: The tag to query.

        Returns:
            List of ``(step, value)`` tuples.
        """
        return list(self._scalar_cache.get(tag, []))

    # -- Context managers for timing -------------------------------------------

    @contextmanager
    def timer(self, name: str, step: int | None = None) -> Generator[None, None, None]:
        """Context manager that logs the elapsed time of a code block.

        Args:
            name: Timer name (logged under ``timing/{name}``).
            step: Step index.

        Yields:
            Nothing. Duration is logged on exit.

        Example::

            with tb.timer("forward_pass", step=100):
                output = model(input)
        """
        t0 = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - t0
            self.log_timing(name, elapsed, step=step)

    @contextmanager
    def train_step_context(self, step: int) -> Generator[dict[str, float], None, None]:
        """Context manager for a training step that collects metrics.

        Yields a dictionary that the caller can populate with metric names and
        values.  On exit, all collected metrics are logged under ``train/``.

        Args:
            step: The training step index.

        Yields:
            Mutable dictionary to be populated with metric values.

        Example::

            with tb.train_step_context(step=42) as metrics:
                loss = compute_loss()
                metrics["loss"] = loss
                metrics["accuracy"] = compute_acc()
        """
        metrics: dict[str, float] = {}
        t0 = time.time()
        try:
            yield metrics
        finally:
            elapsed = time.time() - t0
            metrics["step_time"] = elapsed
            for k, v in metrics.items():
                self.add_scalar(f"train/{k}", v, step=step)
            self._step_tracker.set(step)

    # -- Flush / close ----------------------------------------------------------

    def flush(self) -> None:
        """Flush pending events to disk."""
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Flush and close the TensorBoard writer.

        Safe to call multiple times.
        """
        if self._closed:
            return
        self.flush()
        if self._writer is not None:
            self._writer.close()
        self._closed = True

    @property
    def is_closed(self) -> bool:
        """Whether this logger has been closed."""
        return self._closed


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_tb_logger(
    log_dir: str | Path,
    experiment_name: str | None = None,
    enabled: bool = True,
    **kwargs: Any,
) -> TBLogger:
    """Create a TBLogger, falling back to disabled mode if TensorBoard is
    not installed.

    Args:
        log_dir: Root directory for event files.
        experiment_name: Optional experiment sub-directory name.
        enabled: Whether logging should be active.
        **kwargs: Forwarded to ``TBLogger.__init__``.

    Returns:
        A ``TBLogger`` instance (possibly disabled).
    """
    if enabled and not _TB_AVAILABLE:
        warnings.warn(
            "TensorBoard is not installed; logger will be disabled. "
            "Install with: pip install tensorboard",
            stacklevel=2,
        )
        enabled = False
    return TBLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        enabled=enabled,
        **kwargs,
    )
