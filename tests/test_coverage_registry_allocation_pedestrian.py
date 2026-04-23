"""Targeted coverage tests for registry error paths, task allocation
edge cases, and pedestrian state helpers.

Covers previously-missed lines in:
    navirl/core/registry.py (67-69, 115, 130-132, 178, 193-195, 284-285, 311-312)
    navirl/coordination/task_allocation.py (120, 169, 251-268, 316)
    navirl/humans/pedestrian_state.py (93, 332-333, 421, 431, 444, 472,
                                      487, 497-499, 509-511, 617-636, 673, 683)
"""

from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest

from navirl.coordination.task_allocation import (
    AllocationResult,
    AuctionAllocator,
    GreedyAllocator,
    HungarianAllocator,
    Task,
)
from navirl.core.plugin_validation import PluginValidationError
from navirl.core.registry import (
    _BACKENDS,
    _HUMAN_CONTROLLERS,
    _ROBOT_CONTROLLERS,
    get_plugin_info,
    register_backend,
    register_human_controller,
    register_robot_controller,
    validate_all_plugins,
)
from navirl.humans.pedestrian_state import (
    Activity,
    GazeDirection,
    PedestrianState,
    PersonalityTag,
    StateHistory,
    StatePredictor,
)

# ---------------------------------------------------------------------------
# Registry fixture — snapshot/restore global registries
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries():
    old_b = dict(_BACKENDS)
    old_h = dict(_HUMAN_CONTROLLERS)
    old_r = dict(_ROBOT_CONTROLLERS)
    yield
    _BACKENDS.clear()
    _BACKENDS.update(old_b)
    _HUMAN_CONTROLLERS.clear()
    _HUMAN_CONTROLLERS.update(old_h)
    _ROBOT_CONTROLLERS.clear()
    _ROBOT_CONTROLLERS.update(old_r)


def _factory_fn(config=None):
    return "instance"


# ===========================================================================
# navirl/core/registry.py — error handling and override paths
# ===========================================================================


class TestRegisterBackendErrorPaths:
    """Covers lines 67-69 (PluginValidationError re-raise path)."""

    def test_non_callable_factory_reraises(self, caplog):
        with (
            caplog.at_level("ERROR"),
            pytest.raises(PluginValidationError, match="must be callable"),
        ):
            register_backend("bad_be", "not_a_factory")  # type: ignore[arg-type]
        assert any("Failed to register backend" in r.message for r in caplog.records)

    def test_integer_factory_reraises(self):
        with pytest.raises(PluginValidationError):
            register_backend("bad_int", 42)  # type: ignore[arg-type]


class TestRegisterHumanControllerOverride:
    """Covers line 115 ("Overriding existing human controller")."""

    def test_override_existing_logs_warning(self, caplog):
        register_human_controller("hc_override", _factory_fn)
        with caplog.at_level("WARNING"):
            register_human_controller("hc_override", _factory_fn)
        assert any(
            "Overriding existing human controller" in r.message for r in caplog.records
        )


class TestRegisterHumanControllerErrorPaths:
    """Covers lines 130-132 (PluginValidationError re-raise for human)."""

    def test_non_callable_factory_reraises(self, caplog):
        with (
            caplog.at_level("ERROR"),
            pytest.raises(PluginValidationError, match="must be callable"),
        ):
            register_human_controller("bad_hc", "not_a_factory")  # type: ignore[arg-type]
        assert any(
            "Failed to register human controller" in r.message for r in caplog.records
        )


class TestRegisterRobotControllerOverride:
    """Covers line 178 ("Overriding existing robot controller")."""

    def test_override_existing_logs_warning(self, caplog):
        register_robot_controller("rc_override", _factory_fn)
        with caplog.at_level("WARNING"):
            register_robot_controller("rc_override", _factory_fn)
        assert any(
            "Overriding existing robot controller" in r.message for r in caplog.records
        )


class TestRegisterRobotControllerErrorPaths:
    """Covers lines 193-195 (PluginValidationError re-raise for robot)."""

    def test_non_callable_factory_reraises(self, caplog):
        with (
            caplog.at_level("ERROR"),
            pytest.raises(PluginValidationError, match="must be callable"),
        ):
            register_robot_controller("bad_rc", 3.14)  # type: ignore[arg-type]
        assert any(
            "Failed to register robot controller" in r.message for r in caplog.records
        )


class TestGetPluginInfoSignatureFailure:
    """Covers lines 284-285 (ValueError/TypeError in signature inspection)."""

    def test_signature_valueerror_falls_back_to_unknown(self):
        class _C:
            __navirl_api_version__ = "1.0"

            def __init__(self):
                pass

        # Inject directly to bypass security validation.
        _BACKENDS["sig_fail"] = _C
        with mock.patch(
            "navirl.core.registry.inspect.signature",
            side_effect=ValueError("unsupported callable"),
        ):
            info = get_plugin_info("backend", "sig_fail")
        assert info["init_parameters"] == "unknown"

    def test_signature_typeerror_falls_back_to_unknown(self):
        class _C:
            __navirl_api_version__ = "1.0"

            def __init__(self):
                pass

        _BACKENDS["sig_type_fail"] = _C
        with mock.patch(
            "navirl.core.registry.inspect.signature",
            side_effect=TypeError("bad argument"),
        ):
            info = get_plugin_info("backend", "sig_type_fail")
        assert info["init_parameters"] == "unknown"


class TestValidateAllPluginsFactoryFailure:
    """Covers lines 311-312 (PluginValidationError caught in validate_all_plugins)."""

    def test_invalid_factory_reports_issue(self):
        # Inject a non-callable directly into the registry to bypass
        # registration-time validation. validate_all_plugins should detect
        # the broken factory and report it under "Factory validation:".
        _BACKENDS["broken_fac"] = "not a factory"  # type: ignore[assignment]
        issues = validate_all_plugins()
        key = "backend:broken_fac"
        assert key in issues
        assert any("Factory validation" in msg for msg in issues[key])


# ===========================================================================
# navirl/coordination/task_allocation.py — edge cases
# ===========================================================================


class TestAuctionAllocatorEdgeCases:
    """Covers line 120 (no agent → unassigned) and 169 (bundle no-agent break)."""

    def test_sequential_auction_with_no_agents_all_unassigned(self):
        alloc = AuctionAllocator()
        tasks = [
            Task(id="t1", location=[0.0, 0.0]),
            Task(id="t2", location=[5.0, 0.0]),
        ]
        result = alloc.sequential_auction(agent_positions={}, tasks=tasks)
        assert isinstance(result, AllocationResult)
        assert result.assignments == {}
        assert [t.id for t in result.unassigned] == ["t1", "t2"]
        assert result.total_cost == 0.0

    def test_bundle_auction_with_no_agents_breaks_early(self):
        alloc = AuctionAllocator()
        tasks = [Task(id="t1", location=[0.0, 0.0])]
        result = alloc.bundle_auction(
            agent_positions={}, tasks=tasks, max_bundle_size=3
        )
        assert result.assignments == {}
        # remaining list is returned as unassigned when no bidder breaks the loop
        assert [t.id for t in result.unassigned] == ["t1"]
        assert result.total_cost == 0.0


class TestHungarianAllocatorScipyFallback:
    """Covers lines 251-268 (pure-numpy greedy fallback when scipy is missing)."""

    def test_fallback_solves_assignment_without_scipy(self):
        alloc = HungarianAllocator()
        agents = {"a1": np.array([0.0, 0.0]), "a2": np.array([10.0, 0.0])}
        tasks = [
            Task(id="t1", location=[1.0, 0.0]),
            Task(id="t2", location=[9.0, 0.0]),
        ]

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "scipy.optimize" or name.startswith("scipy"):
                raise ImportError("scipy not available (simulated)")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            result = alloc.allocate(agents, tasks)

        # Both tasks assigned, one per agent, with minimum-cost pairing
        assert len(result.unassigned) == 0
        a1_tasks = [t.id for t in result.assignments["a1"]]
        a2_tasks = [t.id for t in result.assignments["a2"]]
        assert a1_tasks == ["t1"]
        assert a2_tasks == ["t2"]

    def test_solve_fallback_respects_matrix_order(self):
        # Call _solve directly so we exercise the pure-numpy path deterministically
        # independently of scipy availability.
        cost_matrix = np.array(
            [
                [1.0, 5.0],
                [6.0, 2.0],
            ]
        )
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "scipy.optimize":
                raise ImportError("scipy not available (simulated)")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            rows, cols = HungarianAllocator._solve(cost_matrix)

        # Optimal: row 0 -> col 0, row 1 -> col 1
        mapping = dict(zip(rows.tolist(), cols.tolist(), strict=False))
        assert mapping[0] == 0
        assert mapping[1] == 1


class TestGreedyAllocatorSkipsAlreadyAssigned:
    """Covers line 316 (continue when task index was already taken)."""

    def test_duplicate_pair_on_same_task_is_skipped(self):
        # Spacing is chosen so the cost sort interleaves:
        #   (a1→t0)=1.0, (a2→t0)≈5.1, (a1→t1)=100.0, (a2→t1)≈100.1
        # Iteration 1 assigns (a1,t0). Iteration 2 finds j=0 already assigned
        # and hits the `continue` branch before breaking on completion.
        alloc = GreedyAllocator()
        agents = {
            "a1": np.array([0.0, 0.0]),
            "a2": np.array([0.0, 5.0]),
        }
        tasks = [
            Task(id="t0", location=[1.0, 0.0]),
            Task(id="t1", location=[100.0, 0.0]),
        ]
        result = alloc.allocate(agents, tasks)
        assert result.unassigned == []
        # Cheapest assignments pair a1→t0 and a1→t1 (a1 is still cheaper
        # on t1 than a2 is on t1 because distances grow similarly).
        assigned_ids = {tid.id for aid in agents for tid in result.assignments[aid]}
        assert assigned_ids == {"t0", "t1"}


# ===========================================================================
# navirl/humans/pedestrian_state.py — uncovered branches
# ===========================================================================


def _ped(**kw) -> PedestrianState:
    defaults = {
        "pid": 0,
        "position": np.array([0.0, 0.0]),
        "velocity": np.array([0.0, 0.0]),
    }
    defaults.update(kw)
    return PedestrianState(**defaults)


class TestGazeDirectionWrapping:
    """Covers line 93 (d > math.pi branch) of angular_distance."""

    def test_angular_distance_wraps_positive(self):
        # self.angle_rad = -pi+0.1, other_angle = pi-0.1
        # d0 = (pi-0.1) - (-pi+0.1) = 2*pi - 0.2 > pi → subtract 2*pi
        g = GazeDirection(angle_rad=-math.pi + 0.1)
        d = g.angular_distance(math.pi - 0.1)
        # Expected wrapped result: -0.2
        assert d == pytest.approx(-0.2, abs=1e-9)


class TestPedestrianStateFromDictInvalidActivity:
    """Covers lines 332-333 (unknown Activity string falls back to WALKING)."""

    def test_invalid_activity_string_falls_back(self):
        d = {"pid": 3, "activity": "flying", "personality": "normal"}
        state = PedestrianState.from_dict(d)
        assert state.activity == Activity.WALKING
        # personality is also present and valid
        assert state.personality == PersonalityTag.NORMAL


class TestStateHistoryEmptyArrays:
    """Covers lines 364, 421, 431, 444, 472, 487, 497-499, 509-511."""

    def test_capacity_property(self):
        h = StateHistory(capacity=7)
        assert h.capacity == 7

    def test_velocities_array_empty_and_filled(self):
        h = StateHistory(capacity=5)
        vels = h.velocities_array()
        assert vels.shape == (0, 2)
        assert vels.dtype == np.float64
        h.record(_ped(velocity=np.array([1.5, -0.5])), time_s=0.0)
        h.record(_ped(velocity=np.array([0.0, 2.0])), time_s=1.0)
        vels = h.velocities_array()
        assert vels.shape == (2, 2)
        assert vels[0, 0] == pytest.approx(1.5)
        assert vels[1, 1] == pytest.approx(2.0)

    def test_timestamps_array_empty_and_filled(self):
        h = StateHistory(capacity=5)
        empty_ts = h.timestamps_array()
        assert empty_ts.shape == (0,)
        h.record(_ped(), time_s=1.5)
        h.record(_ped(), time_s=2.5)
        ts = h.timestamps_array()
        assert ts.shape == (2,)
        assert ts[0] == pytest.approx(1.5)
        assert ts[1] == pytest.approx(2.5)

    def test_speed_array_empty_and_filled(self):
        h = StateHistory(capacity=5)
        sp = h.speed_array()
        assert sp.shape == (0,)
        h.record(_ped(velocity=np.array([3.0, 4.0])), time_s=0.0)
        sp = h.speed_array()
        assert sp.shape == (1,)
        assert sp[0] == pytest.approx(5.0)

    def test_path_length_single_sample_is_zero(self):
        h = StateHistory(capacity=5)
        h.record(_ped(position=np.array([1.0, 1.0])), time_s=0.0)
        assert h.path_length() == 0.0

    def test_mean_speed_empty_and_filled(self):
        h = StateHistory(capacity=5)
        # speed_array is empty ⇒ mean_speed short-circuits to 0.0
        assert h.mean_speed() == 0.0
        h.record(_ped(velocity=np.array([3.0, 4.0])), time_s=0.0)
        h.record(_ped(velocity=np.array([0.0, 5.0])), time_s=1.0)
        # speeds are 5.0 and 5.0 → mean 5.0
        assert h.mean_speed() == pytest.approx(5.0)

    def test_comfort_array_empty_and_filled(self):
        h = StateHistory(capacity=5)
        assert h.comfort_array().shape == (0,)
        p = _ped()
        p.comfort_level = 0.75
        h.record(p, time_s=0.0)
        arr = h.comfort_array()
        assert arr.shape == (1,)
        assert arr[0] == pytest.approx(0.75)

    def test_stress_array_empty_and_filled(self):
        h = StateHistory(capacity=5)
        assert h.stress_array().shape == (0,)
        p = _ped()
        p.stress_level = 0.25
        h.record(p, time_s=0.0)
        arr = h.stress_array()
        assert arr.shape == (1,)
        assert arr[0] == pytest.approx(0.25)


class TestStatePredictorRegressionAndCollision:
    """Covers lines 617-636 (regression branch), 673 (parallel collision),
    and 683 (collision outside horizon)."""

    def test_predict_from_history_uses_linear_regression(self):
        pred = StatePredictor()
        h = StateHistory(capacity=10)
        # Record a straight-line trajectory: x(t) = t, y(t) = 2t
        for i, t in enumerate([0.0, 0.5, 1.0, 1.5, 2.0]):
            h.record(
                _ped(
                    pid=i,
                    position=np.array([t, 2.0 * t]),
                    velocity=np.array([1.0, 2.0]),
                ),
                time_s=t,
            )
        # Predict 1.0s beyond the last sample (t=3.0)
        predicted = pred.predict_from_history(h, dt=1.0)
        assert predicted is not None
        assert predicted[0] == pytest.approx(3.0, abs=1e-6)
        assert predicted[1] == pytest.approx(6.0, abs=1e-6)

    def test_predict_from_history_truncates_to_last_ten(self):
        pred = StatePredictor()
        h = StateHistory(capacity=30)
        # Inject 20 noisy samples but a clean linear trend in the last 10
        for i in range(10):
            h.record(
                _ped(position=np.array([i * 10.0, 0.0])),
                time_s=i * 10.0,
            )
        for i in range(10, 20):
            t = float(i)
            h.record(
                _ped(position=np.array([t, 2.0 * t])),
                time_s=t,
            )
        # The regression uses the last 10 entries. Prediction at dt=1
        # extrapolates along x(t)=t, y(t)=2t.
        predicted = pred.predict_from_history(h, dt=1.0)
        assert predicted is not None
        assert predicted[0] == pytest.approx(20.0, abs=1e-6)
        assert predicted[1] == pytest.approx(40.0, abs=1e-6)

    def test_collision_time_parallel_overlapping(self):
        """a<1e-12 and c<=0 → agents are overlapping, collision at t=0."""
        pred = StatePredictor()
        # Zero relative velocity, overlapping positions
        a = _ped(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
        )
        a.radius = 0.5
        b = _ped(
            position=np.array([0.5, 0.0]),
            velocity=np.array([1.0, 0.0]),
        )
        b.radius = 0.5
        # Distance = 0.5, combined_r = 1.0, c = 0.25 - 1.0 = -0.75 ≤ 0
        t = pred.collision_time(a, b, max_horizon=10.0)
        assert t == pytest.approx(0.0)

    def test_collision_time_parallel_separated_returns_none(self):
        """a<1e-12 and c>0 → parallel and disjoint ⇒ no collision."""
        pred = StatePredictor()
        a = _ped(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
        )
        a.radius = 0.25
        b = _ped(
            position=np.array([0.0, 10.0]),
            velocity=np.array([1.0, 0.0]),
        )
        b.radius = 0.25
        assert pred.collision_time(a, b, max_horizon=10.0) is None

    def test_collision_time_negative_discriminant_returns_none(self):
        """Non-parallel velocities but trajectories miss → discriminant < 0."""
        pred = StatePredictor()
        # a moves in +x, b moves in +y with a large offset: paths diverge
        a = _ped(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
        )
        a.radius = 0.1
        b = _ped(
            position=np.array([0.0, 10.0]),
            velocity=np.array([0.0, 1.0]),
        )
        b.radius = 0.1
        assert pred.collision_time(a, b, max_horizon=50.0) is None

    def test_collision_time_outside_horizon_returns_none(self):
        """Valid discriminant but both roots outside max_horizon ⇒ None."""
        pred = StatePredictor()
        # Head-on approach with a small horizon; collision happens ~4.5s away
        # but we cap horizon at 1.0s.
        a = _ped(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
        )
        a.radius = 0.25
        b = _ped(
            position=np.array([10.0, 0.0]),
            velocity=np.array([-1.0, 0.0]),
        )
        b.radius = 0.25
        assert pred.collision_time(a, b, max_horizon=1.0) is None
