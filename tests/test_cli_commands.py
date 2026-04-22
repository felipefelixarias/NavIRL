"""Tests for the small CLI command handlers in ``navirl.cli``.

Covers the pure-Python command wrappers that don't need a full simulation:

- ``_cmd_validate`` (scenario validation: valid and invalid paths)
- ``_cmd_eval``     (metrics aggregation over state-log bundles)
- ``_cmd_pack_validate`` (pack manifest validation)

The existing ``test_cli_overseer_layout.py`` is the template for exercising
command handlers via ``build_parser().parse_args([...])`` and ``args.func(args)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from navirl.cli import build_parser


def _write_minimal_bundle(bundle_dir: Path, robot_xy=(0.0, 0.0), goal_xy=(0.01, 0.01)) -> Path:
    """Create a minimal trace bundle (scenario.yaml + state.jsonl) under bundle_dir.

    Returns the path to state.jsonl, matching what ``eval`` accepts as input.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)

    scenario = {
        "horizon": {"dt": 0.1},
        "scene": {"map": {"id": "hallway"}},
    }
    (bundle_dir / "scenario.yaml").write_text(yaml.safe_dump(scenario), encoding="utf-8")

    rx, ry = robot_xy
    gx, gy = goal_xy
    row = {
        "step": 0,
        "agents": [
            {
                "id": 0,
                "kind": "robot",
                "x": rx,
                "y": ry,
                "vx": 0.0,
                "vy": 0.0,
                "radius": 0.3,
                "goal_x": gx,
                "goal_y": gy,
                "behavior": "",
            }
        ],
    }
    state_path = bundle_dir / "state.jsonl"
    state_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return state_path


# ---------------------------------------------------------------------------
# _cmd_validate
# ---------------------------------------------------------------------------


class TestCmdValidate:
    def test_valid_scenario_returns_zero(self, tmp_path: Path, capsys):
        src = Path("navirl/scenarios/library/hallway_pass.yaml")
        scenario_path = tmp_path / "hallway.yaml"
        scenario_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        parser = build_parser()
        args = parser.parse_args(["validate", str(scenario_path)])
        rc = int(args.func(args))

        assert rc == 0
        assert "valid" in capsys.readouterr().out

    def test_invalid_scenario_raises(self, tmp_path: Path):
        # Load a valid scenario then corrupt a field so ``load_scenario``
        # succeeds but ``validate_scenario_dict`` raises.
        src = Path("navirl/scenarios/library/hallway_pass.yaml")
        scenario = yaml.safe_load(src.read_text(encoding="utf-8"))
        scenario["horizon"]["steps"] = -5  # must be a positive integer

        scenario_path = tmp_path / "bad.yaml"
        scenario_path.write_text(yaml.safe_dump(scenario), encoding="utf-8")

        parser = build_parser()
        args = parser.parse_args(["validate", str(scenario_path)])
        with pytest.raises(ValueError, match="validation failed"):
            args.func(args)


# ---------------------------------------------------------------------------
# _cmd_eval
# ---------------------------------------------------------------------------


class TestCmdEval:
    def test_single_bundle_writes_report(self, tmp_path: Path, capsys):
        bundle = tmp_path / "run_0"
        state_path = _write_minimal_bundle(bundle)
        report_dir = tmp_path / "report"

        parser = build_parser()
        args = parser.parse_args(
            ["eval", str(state_path), "--report", str(report_dir)]
        )
        rc = int(args.func(args))

        assert rc == 0
        per_run = json.loads((report_dir / "per_run.json").read_text(encoding="utf-8"))
        assert len(per_run) == 1
        assert per_run[0]["state_path"] == str(state_path)
        assert per_run[0]["bundle_dir"] == str(bundle)
        assert "metrics" in per_run[0]

        aggregate = json.loads((report_dir / "aggregate.json").read_text(encoding="utf-8"))
        assert "avg_success_rate" in aggregate

        report_md = (report_dir / "REPORT.md").read_text(encoding="utf-8")
        assert "# NavIRL Eval Report" in report_md
        assert "num_runs: `1`" in report_md

        # stdout is the absolute path of REPORT.md
        assert str(report_dir / "REPORT.md") in capsys.readouterr().out

    def test_multiple_bundles_aggregate(self, tmp_path: Path):
        s1 = _write_minimal_bundle(tmp_path / "run_a")
        s2 = _write_minimal_bundle(tmp_path / "run_b")
        report_dir = tmp_path / "report"

        parser = build_parser()
        args = parser.parse_args(
            ["eval", str(s1), str(s2), "--report", str(report_dir)]
        )
        rc = int(args.func(args))

        assert rc == 0
        per_run = json.loads((report_dir / "per_run.json").read_text(encoding="utf-8"))
        assert len(per_run) == 2

    def test_no_matching_inputs_raises(self, tmp_path: Path, monkeypatch):
        # ``expand_state_paths`` globs relative to CWD; point it at an empty
        # tmp directory so the relative ``nope`` token resolves to nothing.
        monkeypatch.chdir(tmp_path)
        parser = build_parser()
        args = parser.parse_args(
            ["eval", "nope_does_not_exist", "--report", "out_report"]
        )
        with pytest.raises(FileNotFoundError, match="No state logs"):
            args.func(args)


# ---------------------------------------------------------------------------
# _cmd_pack_validate
# ---------------------------------------------------------------------------


class TestCmdPackValidate:
    def test_builtin_pack(self, capsys):
        manifest_path = Path("navirl/packs/library/social_nav_baseline.yaml")
        assert manifest_path.exists()

        parser = build_parser()
        args = parser.parse_args(["pack", "validate", str(manifest_path)])
        rc = int(args.func(args))

        out = capsys.readouterr().out
        assert rc == 0
        assert "social-nav-baseline" in out
        assert out.rstrip().endswith("valid")
        # Expect at least one scenario listed
        assert "Scenarios:" in out
