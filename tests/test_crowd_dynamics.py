"""Tests for navirl.models.crowd_dynamics — density, flow, pressure, congestion, LoS."""

from __future__ import annotations

import numpy as np
import pytest

from navirl.models.crowd_dynamics import (
    CrowdAnalyzer,
    FundamentalDiagram,
    LevelOfService,
)

# ---------------------------------------------------------------------------
#  CrowdAnalyzer.compute_density
# ---------------------------------------------------------------------------


class TestComputeDensity:
    def test_empty_positions(self):
        grid = CrowdAnalyzer.compute_density(np.empty((0, 2)), (0, 0, 10, 10))
        assert grid.shape[0] > 0
        assert grid.sum() == 0.0

    def test_single_agent(self):
        positions = np.array([[5.0, 5.0]])
        grid = CrowdAnalyzer.compute_density(positions, (0, 0, 10, 10), cell_size=10.0)
        # Single 10x10 cell → density = 1 / 100 = 0.01
        assert grid.shape == (1, 1)
        np.testing.assert_allclose(grid[0, 0], 0.01)

    def test_multiple_agents_same_cell(self):
        positions = np.array([[1.0, 1.0], [1.5, 1.5], [1.9, 1.9]])
        grid = CrowdAnalyzer.compute_density(positions, (0, 0, 10, 10), cell_size=2.0)
        # All three in cell (0, 0) → density = 3 / 4 = 0.75
        assert grid[0, 0] == pytest.approx(0.75)

    def test_agents_at_boundary(self):
        positions = np.array([[0.0, 0.0], [9.99, 9.99]])
        grid = CrowdAnalyzer.compute_density(positions, (0, 0, 10, 10), cell_size=5.0)
        assert grid.shape == (2, 2)
        # One agent in bottom-left, one in top-right
        assert grid[0, 0] > 0
        assert grid[1, 1] > 0


# ---------------------------------------------------------------------------
#  CrowdAnalyzer.compute_flow_field
# ---------------------------------------------------------------------------


class TestComputeFlowField:
    def test_empty(self):
        flow = CrowdAnalyzer.compute_flow_field(
            np.empty((0, 2)), np.empty((0, 2)), (0, 0, 10, 10)
        )
        assert flow.sum() == 0.0

    def test_single_agent(self):
        pos = np.array([[5.0, 5.0]])
        vel = np.array([[1.0, -0.5]])
        flow = CrowdAnalyzer.compute_flow_field(pos, vel, (0, 0, 10, 10), cell_size=10.0)
        np.testing.assert_allclose(flow[0, 0], [1.0, -0.5])

    def test_average_velocity(self):
        pos = np.array([[1.0, 1.0], [1.5, 1.5]])
        vel = np.array([[2.0, 0.0], [0.0, 2.0]])
        flow = CrowdAnalyzer.compute_flow_field(pos, vel, (0, 0, 10, 10), cell_size=10.0)
        np.testing.assert_allclose(flow[0, 0], [1.0, 1.0])


# ---------------------------------------------------------------------------
#  CrowdAnalyzer.compute_crowd_pressure
# ---------------------------------------------------------------------------


class TestComputeCrowdPressure:
    def test_single_agent(self):
        pos = np.array([[0.0, 0.0]])
        vel = np.array([[1.0, 0.0]])
        assert CrowdAnalyzer.compute_crowd_pressure(pos, vel) == 0.0

    def test_two_agents_same_velocity(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        vel = np.array([[1.0, 0.0], [1.0, 0.0]])
        # Same velocity → zero variance → zero pressure
        pressure = CrowdAnalyzer.compute_crowd_pressure(pos, vel, radius=5.0)
        assert pressure == pytest.approx(0.0)

    def test_opposing_velocities_positive_pressure(self):
        # Need 3+ agents within radius so each neighborhood has variance
        pos = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        vel = np.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
        pressure = CrowdAnalyzer.compute_crowd_pressure(pos, vel, radius=5.0)
        assert pressure > 0

    def test_isolated_agents_zero_pressure(self):
        pos = np.array([[0.0, 0.0], [100.0, 100.0]])
        vel = np.array([[1.0, 0.0], [-1.0, 0.0]])
        pressure = CrowdAnalyzer.compute_crowd_pressure(pos, vel, radius=2.0)
        assert pressure == pytest.approx(0.0)


# ---------------------------------------------------------------------------
#  CrowdAnalyzer.detect_congestion
# ---------------------------------------------------------------------------


class TestDetectCongestion:
    def test_no_congestion(self):
        grid = np.array([[0.5, 0.3], [0.1, 0.8]])
        result = CrowdAnalyzer.detect_congestion(grid, threshold=1.7)
        assert result == []

    def test_congested_cells(self):
        grid = np.array([[0.5, 2.0], [1.8, 0.3]])
        result = CrowdAnalyzer.detect_congestion(grid, threshold=1.7)
        assert (0, 1) in result
        assert (1, 0) in result
        assert len(result) == 2


# ---------------------------------------------------------------------------
#  FundamentalDiagram
# ---------------------------------------------------------------------------


class TestFundamentalDiagram:
    def test_zero_density_free_speed(self):
        fd = FundamentalDiagram(v_free=1.34, rho_max=5.4)
        assert fd.speed_from_density(0.0) == pytest.approx(1.34)

    def test_max_density_zero_speed(self):
        fd = FundamentalDiagram(v_free=1.34, rho_max=5.4)
        assert fd.speed_from_density(5.4) == pytest.approx(0.0)

    def test_above_max_density(self):
        fd = FundamentalDiagram(v_free=1.34, rho_max=5.4)
        assert fd.speed_from_density(10.0) == pytest.approx(0.0)

    def test_monotonically_decreasing(self):
        fd = FundamentalDiagram()
        densities = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        speeds = [fd.speed_from_density(d) for d in densities]
        for i in range(len(speeds) - 1):
            assert speeds[i] >= speeds[i + 1]

    def test_flow_from_density(self):
        fd = FundamentalDiagram(v_free=1.34, rho_max=5.4)
        # flow = density * speed
        assert fd.flow_from_density(0.0) == pytest.approx(0.0)  # 0 * v_free
        assert fd.flow_from_density(5.4) == pytest.approx(0.0)  # rho_max * 0

    def test_flow_has_peak(self):
        fd = FundamentalDiagram()
        densities = np.linspace(0.01, fd.rho_max - 0.01, 100)
        flows = [fd.flow_from_density(d) for d in densities]
        assert max(flows) > 0

    def test_array_input(self):
        fd = FundamentalDiagram()
        densities = np.array([0.0, 1.0, 2.0])
        speeds = fd.speed_from_density(densities)
        assert speeds.shape == (3,)

    def test_fit(self):
        fd = FundamentalDiagram()
        # Generate synthetic data from a known model
        densities = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        speeds = np.array([1.34, 1.1, 0.8, 0.5, 0.25, 0.0])
        fd.fit(densities, speeds)
        assert fd.v_free > 0
        assert fd.rho_max > 0

    def test_fit_too_few_points(self):
        fd = FundamentalDiagram(v_free=1.34, rho_max=5.4)
        original_v = fd.v_free
        fd.fit(np.array([1.0]), np.array([1.0]))
        assert fd.v_free == original_v  # should not change


# ---------------------------------------------------------------------------
#  LevelOfService
# ---------------------------------------------------------------------------


class TestLevelOfService:
    def test_grade_a(self):
        assert LevelOfService.classify(0.1) == "A"

    def test_grade_f(self):
        assert LevelOfService.classify(5.0) == "F"

    def test_monotonic_grades(self):
        densities = [0.1, 0.4, 0.7, 1.0, 1.5, 2.0]
        grades = [LevelOfService.classify(d) for d in densities]
        # Grades should be non-decreasing
        grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
        for i in range(len(grades) - 1):
            assert grade_order[grades[i]] <= grade_order[grades[i + 1]]

    def test_evaluate_area_empty(self):
        pos = np.empty((0, 2))
        grades = LevelOfService.evaluate_area(pos, (0, 0, 10, 10), cell_size=5.0)
        assert grades.shape == (2, 2)
        # All empty → grade A
        assert all(g == "A" for g in grades.ravel())

    def test_evaluate_area_dense(self):
        # Pack many agents into a small area
        pos = np.random.default_rng(42).uniform(0, 1, size=(20, 2))
        grades = LevelOfService.evaluate_area(pos, (0, 0, 1, 1), cell_size=1.0)
        # 20 agents in 1m² → 20 ped/m² → grade F
        assert grades[0, 0] == "F"
