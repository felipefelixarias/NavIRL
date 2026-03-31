"""Weights & Biases integration logger for NavIRL.

Provides a high-level interface for logging experiments, hyperparameters,
metrics, artifacts, model checkpoints, trajectory tables, and comparison
charts to Weights & Biases. Handles the optional ``wandb`` dependency
gracefully via try/except.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

try:
    import wandb
    from wandb.sdk.wandb_run import Run as WandbRun

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    WandbRun = None  # type: ignore[assignment,misc]
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_wandb_available() -> bool:
    """Check whether the ``wandb`` package can be imported.

    Returns:
        ``True`` if ``wandb`` is installed and importable.
    """
    return _WANDB_AVAILABLE


# ---------------------------------------------------------------------------
# Sweep configuration helpers
# ---------------------------------------------------------------------------


class SweepConfig:
    """Builder for W&B sweep configurations.

    Provides a fluent interface for defining hyperparameter sweeps.

    Args:
        method: Sweep search strategy (``"grid"``, ``"random"``, or ``"bayes"``).
        metric_name: Name of the metric to optimise.
        metric_goal: Optimisation direction (``"minimize"`` or ``"maximize"``).
        name: Optional human-readable sweep name.

    Example::

        cfg = (
            SweepConfig("bayes", "eval/reward", "maximize")
            .add_uniform("lr", 1e-5, 1e-2)
            .add_categorical("batch_size", [32, 64, 128])
            .build()
        )
    """

    def __init__(
        self,
        method: str = "random",
        metric_name: str = "val_loss",
        metric_goal: str = "minimize",
        name: str | None = None,
    ) -> None:
        self._method = method
        self._metric = {"name": metric_name, "goal": metric_goal}
        self._parameters: dict[str, dict[str, Any]] = {}
        self._name = name
        self._early_terminate: dict[str, Any] | None = None

    def add_uniform(self, name: str, min_val: float, max_val: float) -> SweepConfig:
        """Add a uniform continuous parameter.

        Args:
            name: Parameter name.
            min_val: Minimum value.
            max_val: Maximum value.

        Returns:
            Self for chaining.
        """
        self._parameters[name] = {"distribution": "uniform", "min": min_val, "max": max_val}
        return self

    def add_log_uniform(self, name: str, min_val: float, max_val: float) -> SweepConfig:
        """Add a log-uniform parameter (sampled uniformly in log-space).

        Args:
            name: Parameter name.
            min_val: Minimum value (positive).
            max_val: Maximum value (positive).

        Returns:
            Self for chaining.
        """
        self._parameters[name] = {
            "distribution": "log_uniform_values",
            "min": min_val,
            "max": max_val,
        }
        return self

    def add_categorical(self, name: str, values: list[Any]) -> SweepConfig:
        """Add a categorical parameter.

        Args:
            name: Parameter name.
            values: List of possible values.

        Returns:
            Self for chaining.
        """
        self._parameters[name] = {"values": values}
        return self

    def add_int_uniform(self, name: str, min_val: int, max_val: int) -> SweepConfig:
        """Add a uniform integer parameter.

        Args:
            name: Parameter name.
            min_val: Minimum value.
            max_val: Maximum value.

        Returns:
            Self for chaining.
        """
        self._parameters[name] = {"distribution": "int_uniform", "min": min_val, "max": max_val}
        return self

    def add_constant(self, name: str, value: Any) -> SweepConfig:
        """Add a constant (fixed) parameter.

        Args:
            name: Parameter name.
            value: The fixed value.

        Returns:
            Self for chaining.
        """
        self._parameters[name] = {"value": value}
        return self

    def set_early_terminate(
        self,
        strategy: str = "hyperband",
        min_iter: int = 3,
        eta: int = 3,
        s: int = 2,
    ) -> SweepConfig:
        """Configure early termination for the sweep.

        Args:
            strategy: Termination strategy name.
            min_iter: Minimum number of iterations before termination.
            eta: Down-sampling rate.
            s: Total number of brackets.

        Returns:
            Self for chaining.
        """
        self._early_terminate = {
            "type": strategy,
            "min_iter": min_iter,
            "eta": eta,
            "s": s,
        }
        return self

    def build(self) -> dict[str, Any]:
        """Build the sweep configuration dictionary.

        Returns:
            Dictionary suitable for ``wandb.sweep()``.
        """
        config: dict[str, Any] = {
            "method": self._method,
            "metric": self._metric,
            "parameters": self._parameters,
        }
        if self._name:
            config["name"] = self._name
        if self._early_terminate:
            config["early_terminate"] = self._early_terminate
        return config


# ---------------------------------------------------------------------------
# Alert helper
# ---------------------------------------------------------------------------


class AlertManager:
    """Manages W&B alerts for a run.

    Args:
        wandb_logger: Parent ``WandbLogger`` instance.
    """

    def __init__(self, wandb_logger: WandbLogger) -> None:
        self._logger = wandb_logger

    def send(
        self,
        title: str,
        text: str,
        level: str = "INFO",
        wait_duration: float = 0.0,
    ) -> None:
        """Send an alert.

        Args:
            title: Alert title.
            text: Alert body text.
            level: Severity level (``"INFO"``, ``"WARN"``, ``"ERROR"``).
            wait_duration: Minimum seconds between repeated alerts with the
                same title.
        """
        if not self._logger.enabled:
            return
        level_map = {
            "INFO": wandb.AlertLevel.INFO,
            "WARN": wandb.AlertLevel.WARN,
            "ERROR": wandb.AlertLevel.ERROR,
        }
        alert_level = level_map.get(level.upper(), wandb.AlertLevel.INFO)
        wandb.alert(title=title, text=text, level=alert_level, wait_duration=wait_duration)

    def on_metric_threshold(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        direction: str = "above",
    ) -> None:
        """Send an alert when a metric crosses a threshold.

        Args:
            metric_name: Name of the metric.
            value: Current metric value.
            threshold: Threshold to compare against.
            direction: ``"above"`` or ``"below"``.
        """
        triggered = (direction == "above" and value > threshold) or (
            direction == "below" and value < threshold
        )
        if triggered:
            self.send(
                title=f"Metric alert: {metric_name}",
                text=(f"{metric_name} = {value:.6g} is {direction} " f"threshold {threshold:.6g}"),
                level="WARN",
            )


# ---------------------------------------------------------------------------
# Main logger
# ---------------------------------------------------------------------------


class WandbLogger:
    """Weights & Biases logger for NavIRL experiments.

    Wraps the ``wandb`` SDK and provides convenience methods for logging
    scalars, histograms, tables, artifacts, model checkpoints, trajectory
    visualisations, and hyperparameter configurations.

    Args:
        project: W&B project name.
        entity: W&B team or user entity.
        run_name: Human-readable run name.
        config: Hyperparameter configuration dictionary.
        tags: Optional tags for the run.
        notes: Optional run notes.
        group: Optional experiment group name.
        job_type: Optional job type (e.g. ``"train"``, ``"eval"``).
        mode: W&B mode (``"online"``, ``"offline"``, or ``"disabled"``).
        dir: Local directory for W&B files.
        resume: Resume behaviour (``"allow"``, ``"must"``, ``"never"``, etc.).
        run_id: Explicit run ID for resuming.
        enabled: Master switch. If ``False`` all calls become no-ops.

    Raises:
        ImportError: If *enabled* is ``True`` and *mode* is not ``"disabled"``
            but ``wandb`` is not installed.

    Example::

        with WandbLogger(project="navirl", run_name="ppo_v1") as wb:
            for step in range(1000):
                wb.log({"train/loss": loss}, step=step)
            wb.log_artifact("model.pt", type="model")
    """

    def __init__(
        self,
        project: str = "navirl",
        entity: str | None = None,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        mode: str = "online",
        dir: str | Path | None = None,
        resume: str | None = None,
        run_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled and _WANDB_AVAILABLE
        self._project = project
        self._config = config or {}
        self._step = 0
        self._closed = False
        self._run: Any = None
        self._alert_manager: AlertManager | None = None
        self._custom_charts: dict[str, dict[str, Any]] = {}

        if enabled and not _WANDB_AVAILABLE:
            if mode != "disabled":
                raise ImportError("wandb is not installed. Install with: pip install wandb")
            self._enabled = False

        if self._enabled:
            init_kwargs: dict[str, Any] = {
                "project": project,
                "name": run_name,
                "config": config,
                "tags": tags,
                "notes": notes,
                "group": group,
                "job_type": job_type,
                "mode": mode,
            }
            if entity is not None:
                init_kwargs["entity"] = entity
            if dir is not None:
                init_kwargs["dir"] = str(dir)
            if resume is not None:
                init_kwargs["resume"] = resume
            if run_id is not None:
                init_kwargs["id"] = run_id
            self._run = wandb.init(**init_kwargs)
            self._alert_manager = AlertManager(self)

    # -- Properties -------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether this logger is active."""
        return self._enabled

    @property
    def run(self) -> Any:
        """The underlying ``wandb.Run`` object, or ``None``."""
        return self._run

    @property
    def run_id(self) -> str | None:
        """The W&B run ID, or ``None`` if disabled."""
        if self._run is not None:
            return self._run.id  # type: ignore[union-attr]
        return None

    @property
    def run_url(self) -> str | None:
        """URL of the W&B run page, or ``None``."""
        if self._run is not None:
            return self._run.get_url()  # type: ignore[union-attr]
        return None

    @property
    def alerts(self) -> AlertManager | None:
        """Alert manager for sending W&B alerts."""
        return self._alert_manager

    # -- Context manager --------------------------------------------------------

    def __enter__(self) -> WandbLogger:
        """Enter context manager.

        Returns:
            This logger instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and finish the run."""
        self.finish()

    # -- Config -----------------------------------------------------------------

    def update_config(self, updates: dict[str, Any]) -> None:
        """Update the run's configuration after initialisation.

        Args:
            updates: Key-value pairs to merge into the config.
        """
        if not self._enabled or self._run is None:
            return
        self._config.update(updates)
        self._run.config.update(updates)

    def set_config(self, config: dict[str, Any]) -> None:
        """Replace the run's configuration entirely.

        Args:
            config: New configuration dictionary.
        """
        if not self._enabled or self._run is None:
            return
        self._config = dict(config)
        for k, v in config.items():
            self._run.config[k] = v

    # -- Core logging -----------------------------------------------------------

    def log(
        self,
        data: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Log a dictionary of metrics.

        Args:
            data: Mapping of metric names to values.
            step: Global step. If ``None`` uses the internal counter.
            commit: Whether to commit the step (advance the internal counter).
        """
        if not self._enabled or self._run is None:
            return
        s = step if step is not None else self._step
        self._run.log(data, step=s, commit=commit)
        if commit:
            self._step = s + 1

    def log_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        """Log a single scalar.

        Args:
            tag: Metric name.
            value: Metric value.
            step: Step index.
        """
        self.log({tag: value}, step=step)

    def log_scalars(
        self,
        scalars: dict[str, float],
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        """Log multiple scalars at once.

        Args:
            scalars: Mapping of names to values.
            step: Step index.
            prefix: Optional prefix prepended to each key.
        """
        if prefix:
            scalars = {f"{prefix}/{k}": v for k, v in scalars.items()}
        self.log(scalars, step=step)

    # -- Training helpers -------------------------------------------------------

    def log_training_step(
        self,
        step: int,
        loss: float,
        lr: float | None = None,
        grad_norm: float | None = None,
        extra: dict[str, float] | None = None,
    ) -> None:
        """Log metrics for a training step.

        Args:
            step: Training step.
            loss: Loss value.
            lr: Learning rate.
            grad_norm: Gradient norm.
            extra: Additional metrics.
        """
        data: dict[str, Any] = {"train/loss": loss}
        if lr is not None:
            data["train/lr"] = lr
        if grad_norm is not None:
            data["train/grad_norm"] = grad_norm
        if extra:
            for k, v in extra.items():
                data[f"train/{k}"] = v
        self.log(data, step=step)

    def log_evaluation(
        self,
        step: int,
        metrics: dict[str, float],
        prefix: str = "eval",
    ) -> None:
        """Log evaluation metrics.

        Args:
            step: Evaluation step.
            metrics: Metric name-value pairs.
            prefix: Tag prefix.
        """
        self.log_scalars(metrics, step=step, prefix=prefix)

    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        success: bool | None = None,
        extra: dict[str, float] | None = None,
    ) -> None:
        """Log episode summary metrics.

        Args:
            episode: Episode index.
            reward: Total reward.
            length: Episode length.
            success: Whether episode succeeded.
            extra: Additional metrics.
        """
        data: dict[str, Any] = {
            "episode/reward": reward,
            "episode/length": length,
        }
        if success is not None:
            data["episode/success"] = float(success)
        if extra:
            for k, v in extra.items():
                data[f"episode/{k}"] = v
        self.log(data, step=episode)

    # -- Histograms -------------------------------------------------------------

    def log_histogram(
        self,
        tag: str,
        values: np.ndarray | Sequence[float],
        step: int | None = None,
        num_bins: int = 64,
    ) -> None:
        """Log a histogram.

        Args:
            tag: Metric name.
            values: Array of values.
            step: Step index.
            num_bins: Number of bins.
        """
        if not self._enabled or self._run is None:
            return
        arr = np.asarray(values, dtype=np.float64).ravel()
        histogram = wandb.Histogram(arr, num_bins=num_bins)
        self.log({tag: histogram}, step=step)

    def log_reward_distribution(
        self,
        step: int,
        rewards: np.ndarray | Sequence[float],
        tag: str = "reward/distribution",
    ) -> None:
        """Log reward distribution as a histogram.

        Args:
            step: Step index.
            rewards: Reward values.
            tag: Metric name.
        """
        self.log_histogram(tag, rewards, step=step)

    # -- Tables -----------------------------------------------------------------

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        """Log a W&B Table.

        Args:
            key: Table name.
            columns: Column headers.
            data: List of rows, where each row is a list of values.
            step: Step index.
        """
        if not self._enabled or self._run is None:
            return
        table = wandb.Table(columns=columns, data=data)
        self.log({key: table}, step=step)

    def log_trajectory_table(
        self,
        step: int,
        agent_id: int,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        rewards: np.ndarray | None = None,
        tag: str = "trajectories",
    ) -> None:
        """Log a trajectory as a W&B table.

        Each row represents one timestep of the trajectory.

        Args:
            step: Logging step.
            agent_id: Agent identifier.
            positions: ``(N, 2)`` array of (x, y) positions.
            velocities: Optional ``(N, 2)`` array of (vx, vy) velocities.
            rewards: Optional ``(N,)`` array of per-step rewards.
            tag: Table key prefix.
        """
        if not self._enabled or self._run is None:
            return

        columns = ["timestep", "agent_id", "x", "y"]
        if velocities is not None:
            columns.extend(["vx", "vy", "speed"])
        if rewards is not None:
            columns.append("reward")

        rows: list[list[Any]] = []
        n = len(positions)
        for i in range(n):
            row: list[Any] = [i, agent_id, float(positions[i, 0]), float(positions[i, 1])]
            if velocities is not None:
                vx, vy = float(velocities[i, 0]), float(velocities[i, 1])
                speed = float(np.sqrt(vx**2 + vy**2))
                row.extend([vx, vy, speed])
            if rewards is not None:
                row.append(float(rewards[i]))
            rows.append(row)

        self.log_table(f"{tag}/agent_{agent_id}", columns, rows, step=step)

    # -- Images -----------------------------------------------------------------

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        step: int | None = None,
        caption: str | None = None,
    ) -> None:
        """Log an image.

        Args:
            tag: Image tag.
            image: Image array (HWC, uint8 or float).
            step: Step index.
            caption: Optional caption.
        """
        if not self._enabled or self._run is None:
            return
        wb_img = wandb.Image(image, caption=caption)
        self.log({tag: wb_img}, step=step)

    def log_images(
        self,
        tag: str,
        images: list[np.ndarray],
        step: int | None = None,
        captions: list[str] | None = None,
    ) -> None:
        """Log a batch of images.

        Args:
            tag: Image tag.
            images: List of image arrays.
            step: Step index.
            captions: Optional list of captions.
        """
        if not self._enabled or self._run is None:
            return
        wb_images = []
        for i, img in enumerate(images):
            cap = captions[i] if captions and i < len(captions) else None
            wb_images.append(wandb.Image(img, caption=cap))
        self.log({tag: wb_images}, step=step)

    # -- Artifacts --------------------------------------------------------------

    def log_artifact(
        self,
        path: str | Path,
        name: str | None = None,
        type: str = "dataset",
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """Log a file or directory as a W&B artifact.

        Args:
            path: Path to the file or directory.
            name: Artifact name (defaults to filename).
            type: Artifact type.
            description: Human-readable description.
            metadata: Arbitrary metadata dictionary.
            aliases: Version aliases (e.g. ``["latest", "best"]``).
        """
        if not self._enabled or self._run is None:
            return
        p = Path(path)
        artifact_name = name or p.stem
        artifact = wandb.Artifact(
            artifact_name,
            type=type,
            description=description,
            metadata=metadata or {},
        )
        if p.is_dir():
            artifact.add_dir(str(p))
        else:
            artifact.add_file(str(p))
        self._run.log_artifact(artifact, aliases=aliases or [])

    def log_model_checkpoint(
        self,
        path: str | Path,
        name: str = "model",
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """Log a model checkpoint as a W&B artifact.

        Args:
            path: Path to the checkpoint file.
            name: Artifact name.
            step: Training step (added to metadata).
            metadata: Additional metadata.
            aliases: Version aliases.
        """
        meta = dict(metadata or {})
        if step is not None:
            meta["step"] = step
        alias_list = list(aliases or [])
        if "latest" not in alias_list:
            alias_list.append("latest")
        self.log_artifact(
            path,
            name=name,
            type="model",
            description=f"Model checkpoint at step {step}",
            metadata=meta,
            aliases=alias_list,
        )

    def use_artifact(
        self,
        name: str,
        type: str | None = None,
        alias: str = "latest",
    ) -> Path | None:
        """Download and return the path to an artifact.

        Args:
            name: Artifact name (can include version, e.g. ``"model:v3"``).
            type: Expected artifact type.
            alias: Version alias to use if no version specified in *name*.

        Returns:
            Local path to the downloaded artifact, or ``None`` if disabled.
        """
        if not self._enabled or self._run is None:
            return None
        full_name = name if ":" in name else f"{name}:{alias}"
        artifact = self._run.use_artifact(full_name, type=type)
        return Path(artifact.download())

    # -- Summary ----------------------------------------------------------------

    def set_summary(self, key: str, value: Any) -> None:
        """Set a run summary value.

        Summary values appear in the W&B runs table and are useful for
        final metrics.

        Args:
            key: Summary key.
            value: Summary value.
        """
        if not self._enabled or self._run is None:
            return
        self._run.summary[key] = value

    def set_summaries(self, data: dict[str, Any]) -> None:
        """Set multiple summary values at once.

        Args:
            data: Mapping of keys to values.
        """
        for k, v in data.items():
            self.set_summary(k, v)

    # -- Tags -------------------------------------------------------------------

    def add_tags(self, tags: list[str]) -> None:
        """Add tags to the current run.

        Args:
            tags: List of tag strings.
        """
        if not self._enabled or self._run is None:
            return
        current = list(self._run.tags or [])
        current.extend(tags)
        self._run.tags = tuple(set(current))

    def remove_tags(self, tags: list[str]) -> None:
        """Remove tags from the current run.

        Args:
            tags: Tag strings to remove.
        """
        if not self._enabled or self._run is None:
            return
        current = set(self._run.tags or [])
        current -= set(tags)
        self._run.tags = tuple(current)

    # -- Sweeps -----------------------------------------------------------------

    @staticmethod
    def create_sweep(
        config: dict[str, Any] | SweepConfig,
        project: str = "navirl",
        entity: str | None = None,
    ) -> str | None:
        """Create a W&B sweep and return its ID.

        Args:
            config: Sweep configuration dictionary or ``SweepConfig`` instance.
            project: W&B project name.
            entity: W&B entity.

        Returns:
            Sweep ID string, or ``None`` if ``wandb`` is unavailable.
        """
        if not _WANDB_AVAILABLE:
            logger.warning("wandb not available; cannot create sweep.")
            return None
        if isinstance(config, SweepConfig):
            config = config.build()
        kwargs: dict[str, Any] = {"sweep": config, "project": project}
        if entity:
            kwargs["entity"] = entity
        return wandb.sweep(**kwargs)

    # -- Custom charts ----------------------------------------------------------

    def define_custom_chart(
        self,
        chart_id: str,
        vega_spec: dict[str, Any],
    ) -> None:
        """Register a custom Vega-Lite chart specification.

        The chart can later be logged with :meth:`log_custom_chart`.

        Args:
            chart_id: Unique chart identifier.
            vega_spec: Vega-Lite specification dictionary.
        """
        self._custom_charts[chart_id] = vega_spec

    def log_custom_chart(
        self,
        chart_id: str,
        data: list[dict[str, Any]],
        step: int | None = None,
    ) -> None:
        """Log data for a previously defined custom chart.

        Falls back to a simple table log if the chart has not been defined.

        Args:
            chart_id: Chart identifier matching a prior ``define_custom_chart``.
            data: List of row dictionaries.
            step: Step index.
        """
        if not self._enabled or self._run is None:
            return
        if not data:
            return

        columns = list(data[0].keys())
        rows = [[row.get(c) for c in columns] for row in data]
        table = wandb.Table(columns=columns, data=rows)

        if chart_id in self._custom_charts:
            spec = self._custom_charts[chart_id]
            self.log(
                {
                    chart_id: wandb.plot_table(
                        vega_spec_name="",
                        data_table=table,
                        fields={c: c for c in columns},
                    )
                },
                step=step,
            )
        else:
            self.log({chart_id: table}, step=step)

    # -- Context managers -------------------------------------------------------

    @contextmanager
    def train_step_context(self, step: int) -> Generator[dict[str, float], None, None]:
        """Context manager that collects and logs training metrics.

        Yields a dictionary that the caller populates. On exit all values
        are logged under ``train/``.

        Args:
            step: Training step index.

        Yields:
            Mutable metrics dictionary.

        Example::

            with wb.train_step_context(step=42) as m:
                m["loss"] = compute_loss()
        """
        metrics: dict[str, float] = {}
        t0 = time.time()
        try:
            yield metrics
        finally:
            elapsed = time.time() - t0
            metrics["step_time"] = elapsed
            self.log_scalars(metrics, step=step, prefix="train")

    @contextmanager
    def eval_context(self, step: int) -> Generator[dict[str, float], None, None]:
        """Context manager that collects and logs evaluation metrics.

        Args:
            step: Evaluation step.

        Yields:
            Mutable metrics dictionary.
        """
        metrics: dict[str, float] = {}
        t0 = time.time()
        try:
            yield metrics
        finally:
            elapsed = time.time() - t0
            metrics["eval_time"] = elapsed
            self.log_scalars(metrics, step=step, prefix="eval")

    # -- Watch ------------------------------------------------------------------

    def watch_model(
        self,
        model: Any,
        log: str = "all",
        log_freq: int = 100,
        log_graph: bool = False,
    ) -> None:
        """Watch a model for gradient and parameter logging.

        This is a pass-through to ``wandb.watch``.  The *model* parameter is
        framework-agnostic; ``wandb`` will inspect it at runtime.

        Args:
            model: The model object.
            log: What to log (``"gradients"``, ``"parameters"``, ``"all"``).
            log_freq: Logging frequency in steps.
            log_graph: Whether to log the computation graph.
        """
        if not self._enabled:
            return
        wandb.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    # -- Finish -----------------------------------------------------------------

    def finish(self, exit_code: int | None = None, quiet: bool = False) -> None:
        """Finish the W&B run.

        Args:
            exit_code: Optional process exit code.
            quiet: Suppress finish output.
        """
        if self._closed:
            return
        if self._run is not None:
            kwargs: dict[str, Any] = {}
            if exit_code is not None:
                kwargs["exit_code"] = exit_code
            if quiet:
                kwargs["quiet"] = quiet
            self._run.finish(**kwargs)
        self._closed = True

    @property
    def is_closed(self) -> bool:
        """Whether this logger has been finished."""
        return self._closed


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_wandb_logger(
    project: str = "navirl",
    enabled: bool = True,
    mode: str | None = None,
    **kwargs: Any,
) -> WandbLogger:
    """Create a WandbLogger, falling back to disabled mode if wandb is not
    installed.

    Args:
        project: W&B project name.
        enabled: Whether logging should be active.
        mode: Override W&B mode. Defaults to ``"disabled"`` when wandb is
            unavailable.
        **kwargs: Forwarded to ``WandbLogger.__init__``.

    Returns:
        A ``WandbLogger`` instance (possibly disabled).
    """
    if enabled and not _WANDB_AVAILABLE:
        warnings.warn(
            "wandb is not installed; logger will be disabled. " "Install with: pip install wandb",
            stacklevel=2,
        )
        enabled = False
    if mode is None and not _WANDB_AVAILABLE:
        mode = "disabled"
    return WandbLogger(project=project, enabled=enabled, mode=mode or "online", **kwargs)
