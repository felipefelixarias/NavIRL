"""Coverage-focused tests for PolicyHumanController / PolicyRobotController.

The lazy-loading path in both controllers is gated on a real ``import torch``.
``tests/test_policy_controllers.py`` covers the direct-args / cfg-dict surface
and the goal-reached short-circuit, but leaves the ``_ensure_loaded`` branches
and the full ``step`` ensemble-inference path exercised only when PyTorch is
installed.

These tests use ``monkeypatch`` to inject a fake ``torch`` module into
``sys.modules`` so the file-resolution and ensemble-inference logic can be
reached in every environment — including minimal containers without torch.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from navirl.core.types import Action, AgentState
from navirl.models.learned_policy import PolicyHumanController
from navirl.models.learned_robot_policy import PolicyRobotController

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_state(
    agent_id: int,
    kind: str = "human",
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    goal_x: float = 5.0,
    goal_y: float = 5.0,
    radius: float = 0.18,
    max_speed: float = 0.8,
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        kind=kind,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        goal_x=goal_x,
        goal_y=goal_y,
        radius=radius,
        max_speed=max_speed,
    )


def _noop_emit(event_type, agent_id, payload):
    pass


class _FakeTensor:
    """Minimal tensor stand-in: wraps a numpy array and supports unsqueeze."""

    def __init__(self, data, device=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.device = device

    def unsqueeze(self, _dim):
        return _FakeTensor(self.data[np.newaxis, ...], device=self.device)

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def detach(self):
        return self


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _install_fake_torch(monkeypatch, *, return_vel=(0.1, 0.2)) -> list[_FakeTensor]:
    """Install a minimal fake ``torch`` module into ``sys.modules``.

    Returns a list that captures every tensor produced by ``torch.tensor``
    so tests can inspect what was sent to the fake model.
    """
    received: list[_FakeTensor] = []

    def _tensor(data, dtype=None, device=None):
        t = _FakeTensor(data, device=device)
        received.append(t)
        return t

    def _device(name: str):
        return f"device:{name}"

    fake = types.SimpleNamespace(
        tensor=_tensor,
        float32="float32",
        device=_device,
        no_grad=_FakeNoGrad,
        jit=types.SimpleNamespace(load=lambda *a, **kw: None),
    )

    monkeypatch.setitem(sys.modules, "torch", fake)
    return received


# ---------------------------------------------------------------------------
#  _ensure_loaded — torch-missing branch
# ---------------------------------------------------------------------------


class TestEnsureLoadedImportError:
    """Exercises the ``except ImportError`` branch of ``_ensure_loaded``."""

    def test_human_controller_raises_importerror(self, monkeypatch):
        # Setting sys.modules[...] = None makes "import torch" raise ImportError,
        # regardless of whether the real torch is on the machine.
        monkeypatch.setitem(sys.modules, "torch", None)
        ctrl = PolicyHumanController(cfg={"model_path": "/tmp/model.pt"})
        with pytest.raises(ImportError, match="PyTorch"):
            ctrl._ensure_loaded()

    def test_robot_controller_raises_importerror(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        ctrl = PolicyRobotController(cfg={"model_path": "/tmp/robot.pt"})
        with pytest.raises(ImportError, match="PyTorch"):
            ctrl._ensure_loaded()


# ---------------------------------------------------------------------------
#  _ensure_loaded — empty / missing model_path
# ---------------------------------------------------------------------------


class TestEnsureLoadedFileErrors:
    """Exercises the FileNotFoundError branches in ``_ensure_loaded``."""

    def test_robot_controller_empty_model_path_raises(self, monkeypatch):
        _install_fake_torch(monkeypatch)
        ctrl = PolicyRobotController()  # defaults to empty model_path_str
        with pytest.raises(FileNotFoundError, match="model_path"):
            ctrl._ensure_loaded()

    def test_human_controller_nonexistent_file_raises(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        ctrl = PolicyHumanController(cfg={"model_path": str(tmp_path / "missing.pt")})
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            ctrl._ensure_loaded()

    def test_robot_controller_nonexistent_file_raises(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        ctrl = PolicyRobotController(
            cfg={"model_path": str(tmp_path / "missing.pt")}
        )
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            ctrl._ensure_loaded()

    def test_human_controller_empty_directory_raises(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        ctrl = PolicyHumanController(cfg={"model_path": str(empty_dir)})
        with pytest.raises(FileNotFoundError, match=r"No \.pt/\.pth files"):
            ctrl._ensure_loaded()

    def test_robot_controller_empty_directory_raises(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        empty_dir = tmp_path / "empty_robot"
        empty_dir.mkdir()
        ctrl = PolicyRobotController(cfg={"model_path": str(empty_dir)})
        with pytest.raises(FileNotFoundError, match=r"No \.pt/\.pth files"):
            ctrl._ensure_loaded()


# ---------------------------------------------------------------------------
#  _ensure_loaded — successful loading
# ---------------------------------------------------------------------------


class TestEnsureLoadedSuccess:
    def test_loads_single_file(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        # Replace the loader so the fake torch can "open" our dummy file.
        loaded_paths: list[str] = []

        def _fake_load(path, map_location=None):
            loaded_paths.append(str(path))
            return types.SimpleNamespace(eval=lambda: None)

        sys.modules["torch"].jit.load = _fake_load  # type: ignore[attr-defined]

        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"placeholder")

        ctrl = PolicyHumanController(cfg={"model_path": str(model_file)})
        ctrl._ensure_loaded()

        assert ctrl._loaded is True
        assert len(ctrl._models) == 1
        assert loaded_paths == [str(model_file)]

    def test_loads_ensemble_from_directory(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        loaded_paths: list[str] = []

        def _fake_load(path, map_location=None):
            loaded_paths.append(str(path))
            return types.SimpleNamespace(eval=lambda: None)

        sys.modules["torch"].jit.load = _fake_load  # type: ignore[attr-defined]

        model_dir = tmp_path / "ensemble"
        model_dir.mkdir()
        # Mix .pt and .pth files to exercise both globs.
        (model_dir / "a.pt").write_bytes(b"a")
        (model_dir / "b.pt").write_bytes(b"b")
        (model_dir / "c.pth").write_bytes(b"c")

        ctrl = PolicyRobotController(cfg={"model_path": str(model_dir)})
        ctrl._ensure_loaded()

        assert ctrl._loaded is True
        assert len(ctrl._models) == 3
        # All three files should have been handed to the loader.
        assert {p.rsplit("/", 1)[-1] for p in loaded_paths} == {"a.pt", "b.pt", "c.pth"}

    def test_robot_loads_single_file(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        loaded_paths: list[str] = []

        def _fake_load(path, map_location=None):
            loaded_paths.append(str(path))
            return types.SimpleNamespace(eval=lambda: None)

        sys.modules["torch"].jit.load = _fake_load  # type: ignore[attr-defined]

        model_file = tmp_path / "robot_model.pt"
        model_file.write_bytes(b"placeholder")

        ctrl = PolicyRobotController(cfg={"model_path": str(model_file)})
        ctrl._ensure_loaded()

        assert ctrl._loaded is True
        assert len(ctrl._models) == 1
        assert loaded_paths == [str(model_file)]

    def test_ensure_loaded_is_idempotent(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        calls: list[str] = []

        def _fake_load(path, map_location=None):
            calls.append(str(path))
            return types.SimpleNamespace(eval=lambda: None)

        sys.modules["torch"].jit.load = _fake_load  # type: ignore[attr-defined]

        model_file = tmp_path / "m.pt"
        model_file.write_bytes(b"x")
        ctrl = PolicyHumanController(cfg={"model_path": str(model_file)})

        ctrl._ensure_loaded()
        ctrl._ensure_loaded()  # second call should hit the early-return guard
        assert len(calls) == 1


# ---------------------------------------------------------------------------
#  step() — full ensemble inference
# ---------------------------------------------------------------------------


class _FakeModel:
    """Scripted model that returns a fixed ``(vx, vy)`` velocity per forward."""

    def __init__(self, vx: float, vy: float):
        self.vx = vx
        self.vy = vy
        self.calls: list[_FakeTensor] = []

    def __call__(self, obs_tensor: _FakeTensor) -> _FakeTensor:
        self.calls.append(obs_tensor)
        return _FakeTensor(np.array([[self.vx, self.vy]], dtype=np.float32))


class TestHumanControllerStep:
    def test_step_averages_ensemble_outputs(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        ctrl = PolicyHumanController(
            cfg={"model_path": str(tmp_path / "m.pt"), "max_neighbours": 0}
        )

        # Pretend the ensemble is already loaded with two scripted models.
        ctrl._loaded = True
        ctrl._device = "device:cpu"
        m1, m2 = _FakeModel(0.2, 0.0), _FakeModel(0.4, 0.0)
        ctrl._models = [m1, m2]

        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (10.0, 0.0)},  # far away -> no goal swap
        )

        actions = ctrl.step(
            0, 0.0, 0.1, {1: _make_state(1, x=1.0, y=0.0)}, robot_id=99, emit_event=_noop_emit
        )

        assert set(actions.keys()) == {1}
        act = actions[1]
        assert isinstance(act, Action)
        assert act.behavior == "LEARNED"
        # Mean of (0.2, 0.4) = 0.3
        assert abs(act.pref_vx - 0.3) < 1e-6
        assert abs(act.pref_vy) < 1e-6
        assert act.metadata["ensemble_size"] == 2
        # Both models were called exactly once.
        assert len(m1.calls) == 1 and len(m2.calls) == 1

    def test_step_clamps_to_max_speed(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        ctrl = PolicyHumanController(
            cfg={"model_path": str(tmp_path / "m.pt"), "max_neighbours": 0}
        )
        ctrl._loaded = True
        ctrl._device = "device:cpu"
        # Model emits a velocity of magnitude 5.0, far above max_speed.
        ctrl._models = [_FakeModel(3.0, 4.0)]
        ctrl.reset(human_ids=[1], starts={1: (0, 0)}, goals={1: (10, 10)})

        state = _make_state(1, x=0.0, y=0.0, max_speed=1.0)
        actions = ctrl.step(0, 0.0, 0.1, {1: state}, robot_id=99, emit_event=_noop_emit)
        act = actions[1]

        speed = math.hypot(act.pref_vx, act.pref_vy)
        assert abs(speed - 1.0) < 1e-6  # clamped to max_speed
        # Direction should be preserved.
        assert abs(act.pref_vx - 3.0 / 5.0) < 1e-6
        assert abs(act.pref_vy - 4.0 / 5.0) < 1e-6
        assert act.metadata["raw_speed"] == pytest.approx(5.0)

    def test_step_swaps_goal_on_arrival(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        ctrl = PolicyHumanController(
            cfg={"model_path": str(tmp_path / "m.pt"), "max_neighbours": 0}
        )
        ctrl._loaded = True
        ctrl._device = "device:cpu"
        ctrl._models = [_FakeModel(0.1, 0.0)]

        start, goal = (0.0, 0.0), (1.0, 0.0)
        ctrl.reset(human_ids=[1], starts={1: start}, goals={1: goal})

        events: list[tuple[str, int, dict]] = []

        def record(et, aid, payload):
            events.append((et, aid, payload))

        state = _make_state(1, x=1.0, y=0.0, goal_x=1.0, goal_y=0.0)
        ctrl.step(0, 0.0, 0.1, {1: state}, robot_id=99, emit_event=record)

        # Goal should be swapped because we're within 0.5m of the goal.
        assert ctrl.goals[1] == start
        assert ctrl.starts[1] == goal
        assert events and events[0][0] == "goal_swap"
        assert events[0][1] == 1

    def test_step_honours_neighbour_count(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        ctrl = PolicyHumanController(
            cfg={"model_path": str(tmp_path / "m.pt"), "max_neighbours": 2}
        )
        ctrl._loaded = True
        ctrl._device = "device:cpu"
        recorder = _FakeModel(0.0, 0.0)
        ctrl._models = [recorder]
        ctrl.reset(human_ids=[1], starts={1: (0, 0)}, goals={1: (9, 9)})

        states = {
            1: _make_state(1, x=0.0, y=0.0),
            2: _make_state(2, x=1.0, y=0.0),
            3: _make_state(3, x=2.0, y=0.0),
        }
        ctrl.step(0, 0.0, 0.1, states, robot_id=99, emit_event=_noop_emit)

        # Obs layout: own (6) + max_neighbours=2 * 5 = 16 total.
        assert recorder.calls[0].data.shape == (1, 16)


class TestRobotControllerStep:
    def test_step_averages_ensemble_outputs(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        ctrl = PolicyRobotController(
            cfg={
                "model_path": str(tmp_path / "m.pt"),
                "max_neighbours": 0,
                "goal_tolerance": 0.01,
                "max_speed": 2.0,
            }
        )
        ctrl._loaded = True
        ctrl._device = "device:cpu"
        m1, m2 = _FakeModel(0.2, 0.0), _FakeModel(0.6, 0.0)
        ctrl._models = [m1, m2]
        ctrl.reset(robot_id=0, start=(0, 0), goal=(10, 0), backend=None)

        robot = _make_state(0, kind="robot", x=1.0, y=0.0, max_speed=2.0)
        action = ctrl.step(0, 0.0, 0.1, {0: robot}, _noop_emit)

        assert action.behavior == "LEARNED"
        assert abs(action.pref_vx - 0.4) < 1e-6  # mean of (0.2, 0.6)
        assert abs(action.pref_vy) < 1e-6
        assert action.metadata["ensemble_size"] == 2

    def test_step_clamps_to_effective_max_speed(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        ctrl = PolicyRobotController(
            cfg={
                "model_path": str(tmp_path / "m.pt"),
                "max_neighbours": 0,
                "goal_tolerance": 0.01,
                "max_speed": 0.5,  # tighter of the two caps
            }
        )
        ctrl._loaded = True
        ctrl._device = "device:cpu"
        ctrl._models = [_FakeModel(3.0, 4.0)]  # magnitude 5.0
        ctrl.reset(robot_id=0, start=(0, 0), goal=(10, 0), backend=None)

        robot = _make_state(0, kind="robot", x=1.0, y=0.0, max_speed=1.5)
        action = ctrl.step(0, 0.0, 0.1, {0: robot}, _noop_emit)

        speed = math.hypot(action.pref_vx, action.pref_vy)
        # Effective max = min(robot.max_speed=1.5, cfg.max_speed=0.5) = 0.5
        assert abs(speed - 0.5) < 1e-6

    def test_step_includes_human_flag_in_observation(self, monkeypatch, tmp_path):
        _install_fake_torch(monkeypatch)
        ctrl = PolicyRobotController(
            cfg={
                "model_path": str(tmp_path / "m.pt"),
                "max_neighbours": 2,
                "goal_tolerance": 0.01,
            }
        )
        ctrl._loaded = True
        ctrl._device = "device:cpu"
        recorder = _FakeModel(0.0, 0.0)
        ctrl._models = [recorder]
        ctrl.reset(robot_id=0, start=(0, 0), goal=(10, 0), backend=None)

        states = {
            0: _make_state(0, kind="robot"),
            1: _make_state(1, kind="human", x=1.0, y=0.0),
            2: _make_state(2, kind="robot", x=2.0, y=0.0),
        }
        ctrl.step(0, 0.0, 0.1, states, _noop_emit)

        # Obs layout: own (6) + max_neighbours=2 * 6 = 18.
        obs = recorder.calls[0].data[0]
        assert obs.shape == (18,)
        # First-neighbour kind flag (offset 6 + 5) is human → 1.0.
        assert abs(obs[6 + 5] - 1.0) < 1e-6
        # Second-neighbour kind flag (offset 12 + 5) is robot → 0.0.
        assert abs(obs[12 + 5]) < 1e-6
