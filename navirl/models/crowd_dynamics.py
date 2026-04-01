"""Macroscopic crowd analysis tools.

Provides density estimation, flow-field computation, crowd pressure metrics,
congestion detection, fundamental diagram fitting, and level-of-service
classification based on the Fruin framework.
"""

from __future__ import annotations

import numpy as np

from navirl.core.constants import (
    FUNDAMENTAL_DIAGRAM_RHO_MAX,
    FUNDAMENTAL_DIAGRAM_V_FREE,
)
from navirl.core.constants import (
    LOS as LOS_CONSTANTS,
)

__all__ = ["CrowdAnalyzer", "FundamentalDiagram", "LevelOfService"]


# ---------------------------------------------------------------------------
#  CrowdAnalyzer – spatial density, flow, pressure, and congestion
# ---------------------------------------------------------------------------


class CrowdAnalyzer:
    """Macroscopic crowd metrics computed on a spatial grid."""

    # -- density --------------------------------------------------------

    @staticmethod
    def compute_density(
        positions: np.ndarray,
        region_bounds: tuple[float, float, float, float],
        cell_size: float = 1.0,
    ) -> np.ndarray:
        """Compute a 2-D density grid (pedestrians per m²).

        Parameters
        ----------
        positions:
            ``(N, 2)`` array of agent *(x, y)* positions.
        region_bounds:
            ``(x_min, y_min, x_max, y_max)`` of the analysis region.
        cell_size:
            Side length (metres) of each grid cell.

        Returns
        -------
        np.ndarray
            2-D array of densities (ped/m²) with shape
            ``(n_rows, n_cols)``.
        """
        x_min, y_min, x_max, y_max = region_bounds
        n_cols = max(1, int(np.ceil((x_max - x_min) / cell_size)))
        n_rows = max(1, int(np.ceil((y_max - y_min) / cell_size)))

        grid = np.zeros((n_rows, n_cols), dtype=np.float64)

        if len(positions) == 0:
            return grid

        # Bin each agent into its grid cell.
        col_idx = np.clip(((positions[:, 0] - x_min) / cell_size).astype(int), 0, n_cols - 1)
        row_idx = np.clip(((positions[:, 1] - y_min) / cell_size).astype(int), 0, n_rows - 1)

        np.add.at(grid, (row_idx, col_idx), 1)

        cell_area = cell_size * cell_size
        grid /= cell_area
        return grid

    # -- flow field -----------------------------------------------------

    @staticmethod
    def compute_flow_field(
        positions: np.ndarray,
        velocities: np.ndarray,
        region_bounds: tuple[float, float, float, float],
        cell_size: float = 1.0,
    ) -> np.ndarray:
        """Compute an average velocity (flow) field over a spatial grid.

        Parameters
        ----------
        positions:
            ``(N, 2)`` array of agent positions.
        velocities:
            ``(N, 2)`` array of agent velocities.
        region_bounds:
            ``(x_min, y_min, x_max, y_max)``.
        cell_size:
            Grid cell side length (metres).

        Returns
        -------
        np.ndarray
            ``(n_rows, n_cols, 2)`` array of mean velocity vectors per cell.
        """
        x_min, y_min, x_max, y_max = region_bounds
        n_cols = max(1, int(np.ceil((x_max - x_min) / cell_size)))
        n_rows = max(1, int(np.ceil((y_max - y_min) / cell_size)))

        flow = np.zeros((n_rows, n_cols, 2), dtype=np.float64)
        counts = np.zeros((n_rows, n_cols), dtype=np.float64)

        if len(positions) == 0:
            return flow

        col_idx = np.clip(((positions[:, 0] - x_min) / cell_size).astype(int), 0, n_cols - 1)
        row_idx = np.clip(((positions[:, 1] - y_min) / cell_size).astype(int), 0, n_rows - 1)

        np.add.at(flow[:, :, 0], (row_idx, col_idx), velocities[:, 0])
        np.add.at(flow[:, :, 1], (row_idx, col_idx), velocities[:, 1])
        np.add.at(counts, (row_idx, col_idx), 1)

        mask = counts > 0
        flow[mask, 0] /= counts[mask]
        flow[mask, 1] /= counts[mask]

        return flow

    # -- crowd pressure -------------------------------------------------

    @staticmethod
    def compute_crowd_pressure(
        positions: np.ndarray,
        velocities: np.ndarray,
        radius: float = 2.0,
    ) -> float:
        """Estimate crowd pressure as per Helbing et al.

        Crowd pressure is the variance of velocity directions within a
        local neighbourhood, weighted by local density.  High pressure
        indicates turbulent crowd motion and is a precursor to crush
        events.

        Parameters
        ----------
        positions:
            ``(N, 2)`` agent positions.
        velocities:
            ``(N, 2)`` agent velocities.
        radius:
            Neighbourhood radius (metres) for the local estimate.

        Returns
        -------
        float
            Scalar crowd-pressure metric (>= 0).
        """
        n = len(positions)
        if n < 2:
            return 0.0

        pressure_sum = 0.0
        for i in range(n):
            diffs = positions - positions[i]
            dists = np.linalg.norm(diffs, axis=1)
            neighbours = (dists < radius) & (dists > 0)
            n_neighbours = int(np.sum(neighbours))
            if n_neighbours == 0:
                continue

            local_density = n_neighbours / (np.pi * radius * radius)
            local_vels = velocities[neighbours]
            vel_variance = float(np.var(local_vels, axis=0).sum())
            pressure_sum += local_density * vel_variance

        return pressure_sum / n

    # -- congestion detection -------------------------------------------

    @staticmethod
    def detect_congestion(
        density_grid: np.ndarray,
        threshold: float = 1.7,
    ) -> list[tuple[int, int]]:
        """Identify grid cells exceeding a density threshold.

        Parameters
        ----------
        density_grid:
            ``(n_rows, n_cols)`` density array (ped/m²).
        threshold:
            Density above which a cell is considered congested.
            Default ``1.7`` corresponds to Fruin LoS F.

        Returns
        -------
        list[tuple[int, int]]
            List of ``(row, col)`` indices of congested cells.
        """
        rows, cols = np.where(density_grid >= threshold)
        return list(zip(rows.tolist(), cols.tolist(), strict=False))


# ---------------------------------------------------------------------------
#  FundamentalDiagram – speed-density relationship
# ---------------------------------------------------------------------------


class FundamentalDiagram:
    """Speed-density fundamental diagram using the Weidmann model.

    The default relationship is:

        v(rho) = v_free * (1 - exp(-gamma * (1/rho - 1/rho_max)))

    For the simpler linear variant:

        v(rho) = v_free * max(0, 1 - rho / rho_max)
    """

    def __init__(
        self,
        v_free: float = FUNDAMENTAL_DIAGRAM_V_FREE,
        rho_max: float = FUNDAMENTAL_DIAGRAM_RHO_MAX,
    ) -> None:
        self.v_free = v_free
        self.rho_max = rho_max

    # -- speed from density ---------------------------------------------

    def speed_from_density(self, density: float | np.ndarray) -> float | np.ndarray:
        """Return the expected walking speed for a given density.

        Uses the linear Weidmann approximation:
        ``v = v_free * max(0, 1 - rho / rho_max)``.
        """
        density = np.asarray(density, dtype=np.float64)
        speed = self.v_free * np.clip(1.0 - density / self.rho_max, 0.0, 1.0)
        return float(speed) if speed.ndim == 0 else speed

    # -- flow from density ----------------------------------------------

    def flow_from_density(self, density: float | np.ndarray) -> float | np.ndarray:
        """Return pedestrian flow rate (ped/m/s) = density * speed."""
        density = np.asarray(density, dtype=np.float64)
        speed = self.speed_from_density(density)
        flow = density * speed
        return float(flow) if np.ndim(flow) == 0 else flow

    # -- calibration ----------------------------------------------------

    def fit(
        self,
        density_samples: np.ndarray,
        speed_samples: np.ndarray,
    ) -> None:
        """Calibrate *v_free* and *rho_max* from empirical data.

        Uses ordinary least squares on the linear model
        ``v = v_free * (1 - rho / rho_max)``.

        Parameters
        ----------
        density_samples:
            1-D array of density observations (ped/m²).
        speed_samples:
            1-D array of corresponding speed observations (m/s).
        """
        density_samples = np.asarray(density_samples, dtype=np.float64).ravel()
        speed_samples = np.asarray(speed_samples, dtype=np.float64).ravel()

        if len(density_samples) < 2:
            return

        # v = v_free - (v_free / rho_max) * rho  →  v = a + b * rho
        # Solve via least-squares: [1, rho] @ [a, b]^T = v
        A = np.column_stack([np.ones_like(density_samples), density_samples])
        result, *_ = np.linalg.lstsq(A, speed_samples, rcond=None)
        a, b = result

        if a > 0:
            self.v_free = float(a)
        if b < 0 and a > 0:
            self.rho_max = float(-a / b)


# ---------------------------------------------------------------------------
#  LevelOfService – Fruin walkway LoS classification
# ---------------------------------------------------------------------------


class LevelOfService:
    """Fruin level-of-service classifier for pedestrian areas.

    Grades range from *A* (free flow, < 0.3 ped/m²) to *F*
    (breakdown / crush risk, >= 1.7 ped/m²).
    """

    # Thresholds are sourced from the constants module (LOS dataclass).
    _THRESHOLDS: list[tuple[str, float]] = [
        ("A", LOS_CONSTANTS.A_max_density),
        ("B", LOS_CONSTANTS.B_max_density),
        ("C", LOS_CONSTANTS.C_max_density),
        ("D", LOS_CONSTANTS.D_max_density),
        ("E", LOS_CONSTANTS.E_max_density),
    ]

    # -- classify a single density value --------------------------------

    @classmethod
    def classify(cls, density: float) -> str:
        """Return a LoS grade (``'A'`` – ``'F'``) for a density value.

        Parameters
        ----------
        density:
            Local pedestrian density in ped/m².

        Returns
        -------
        str
            One of ``'A'``, ``'B'``, ``'C'``, ``'D'``, ``'E'``, ``'F'``.
        """
        for grade, max_density in cls._THRESHOLDS:
            if density < max_density:
                return grade
        return "F"

    # -- evaluate an area -----------------------------------------------

    @classmethod
    def evaluate_area(
        cls,
        positions: np.ndarray,
        bounds: tuple[float, float, float, float],
        cell_size: float = 1.0,
    ) -> np.ndarray:
        """Compute per-cell LoS grades for a spatial region.

        Parameters
        ----------
        positions:
            ``(N, 2)`` agent positions.
        bounds:
            ``(x_min, y_min, x_max, y_max)``.
        cell_size:
            Grid cell side length (metres).

        Returns
        -------
        np.ndarray
            2-D object array of single-character grade strings.
        """
        density_grid = CrowdAnalyzer.compute_density(positions, bounds, cell_size)
        n_rows, n_cols = density_grid.shape
        grades = np.empty((n_rows, n_cols), dtype=object)
        for r in range(n_rows):
            for c in range(n_cols):
                grades[r, c] = cls.classify(float(density_grid[r, c]))
        return grades
