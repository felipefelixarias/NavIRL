"""Abstract base classes for sensor simulation.

Provides :class:`SensorBase` (the interface every sensor must implement) and a
family of :class:`NoiseModel` subclasses that can be composed with any sensor
to inject realistic measurement noise.

All numeric operations use **numpy** for vectorised performance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
#  Sensor base
# ---------------------------------------------------------------------------


class SensorBase(ABC):
    """Abstract interface for all simulated sensors.

    Parameters
    ----------
    config : Any
        Sensor-specific configuration dataclass.
    noise_model : NoiseModel | None
        Optional noise model applied to every observation.
    """

    def __init__(self, config: Any, noise_model: NoiseModel | None = None) -> None:
        self.config = config
        self.noise_model = noise_model
        self._rng: np.random.Generator = np.random.default_rng()

    # -- public API ----------------------------------------------------------

    def observe(self, world_state: dict[str, Any]) -> Any:
        """Return a (possibly noisy) observation of *world_state*.

        Subclasses implement :meth:`_raw_observe` which returns the clean
        reading; noise is applied automatically when a noise model is set.
        """
        clean = self._raw_observe(world_state)
        if self.noise_model is not None:
            return self.noise_model.apply(clean)
        return clean

    def reset(self) -> None:
        """Reset any internal state (e.g. temporal buffers).

        Override in subclasses that maintain state across steps.
        """
        pass

    def seed(self, seed: int) -> None:
        """Seed the sensor's random number generator."""
        self._rng = np.random.default_rng(seed)
        if self.noise_model is not None:
            self.noise_model.seed(seed)

    @abstractmethod
    def get_observation_space(self) -> dict[str, Any]:
        """Return a gymnasium.spaces.Space-like dictionary describing the
        observation shape, dtype, and bounds.

        Returns
        -------
        dict
            Keys ``shape``, ``dtype``, ``low``, ``high`` at minimum.
        """
        ...

    # -- private -------------------------------------------------------------

    @abstractmethod
    def _raw_observe(self, world_state: dict[str, Any]) -> Any:
        """Produce a clean (noise-free) observation of *world_state*."""
        ...


# ---------------------------------------------------------------------------
#  Noise model hierarchy
# ---------------------------------------------------------------------------


class NoiseModel(ABC):
    """Base class for sensor noise injection.

    All noise models operate element-wise on numpy arrays and return an array
    of the same shape and dtype.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def seed(self, seed: int) -> None:
        """Re-seed the internal RNG."""
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def apply(self, clean_data: np.ndarray) -> np.ndarray:
        """Return a noisy copy of *clean_data*."""
        ...


# ---------------------------------------------------------------------------
#  Concrete noise models
# ---------------------------------------------------------------------------


@dataclass
class GaussianNoise(NoiseModel):
    """Additive zero-mean Gaussian noise.

    Parameters
    ----------
    std : float
        Standard deviation of the additive noise.
    mean : float
        Mean of the additive noise (default 0).
    """

    std: float = 0.01
    mean: float = 0.0

    def __init__(self, std: float = 0.01, mean: float = 0.0, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.std = std
        self.mean = mean

    def apply(self, clean_data: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(loc=self.mean, scale=self.std, size=clean_data.shape)
        return clean_data + noise.astype(clean_data.dtype)


@dataclass
class SaltPepperNoise(NoiseModel):
    """Salt-and-pepper noise (random extreme outliers).

    With probability *prob*, each element is replaced by either
    ``low`` or ``high`` with equal chance.

    Parameters
    ----------
    prob : float
        Probability that any single element is corrupted.
    low : float
        "Pepper" replacement value.
    high : float
        "Salt" replacement value.
    """

    prob: float = 0.01
    low: float = 0.0
    high: float = 1.0

    def __init__(
        self, prob: float = 0.01, low: float = 0.0, high: float = 1.0, seed: int | None = None
    ) -> None:
        super().__init__(seed=seed)
        self.prob = prob
        self.low = low
        self.high = high

    def apply(self, clean_data: np.ndarray) -> np.ndarray:
        out = clean_data.copy()
        mask = self._rng.random(clean_data.shape) < self.prob
        salt = self._rng.random(clean_data.shape) < 0.5
        out[mask & salt] = self.high
        out[mask & ~salt] = self.low
        return out


@dataclass
class DropoutNoise(NoiseModel):
    """Random dropout (missing values).

    With probability *prob*, each element is replaced by *fill_value*
    (default ``NaN``).

    Parameters
    ----------
    prob : float
        Dropout probability per element.
    fill_value : float
        Value written into dropped-out elements.
    """

    prob: float = 0.02
    fill_value: float = float("nan")

    def __init__(
        self, prob: float = 0.02, fill_value: float = float("nan"), seed: int | None = None
    ) -> None:
        super().__init__(seed=seed)
        self.prob = prob
        self.fill_value = fill_value

    def apply(self, clean_data: np.ndarray) -> np.ndarray:
        out = clean_data.copy()
        mask = self._rng.random(clean_data.shape) < self.prob
        out[mask] = self.fill_value
        return out


@dataclass
class QuantizationNoise(NoiseModel):
    """Quantization noise caused by finite sensor resolution.

    Values are rounded to the nearest multiple of *step*.

    Parameters
    ----------
    step : float
        Quantization step size.
    """

    step: float = 0.01

    def __init__(self, step: float = 0.01, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.step = step

    def apply(self, clean_data: np.ndarray) -> np.ndarray:
        return np.round(clean_data / self.step) * self.step
