from __future__ import annotations

import json
from pathlib import Path

import yaml

from navirl.tune import run_tuning


def test_tuning_smoke_single_scenario(tmp_path: Path):
    src = Path("navirl/scenarios/library/hallway_pass.yaml")
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    raw["horizon"]["steps"] = 28
    raw.setdefault("render", {})["enabled"] = True
    raw["render"]["video"] = False

    scenario_path = tmp_path / "hallway_tune.yaml"
    scenario_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    out_dir = tmp_path / "tune"
    result = run_tuning(
        out_root=out_dir,
        suite="quick",
        scenarios=[str(scenario_path)],
        trials=2,
        seed=3,
        judge_mode="heuristic",
        judge_confidence_min=0.6,
        max_frames=6,
        video=False,
    )

    report_path = Path(result["report_path"])
    best_path = Path(result["best_params_path"])
    trials_path = Path(result["trials_path"])
    rerank_path = Path(result["aegis_rerank_path"])

    assert report_path.exists()
    assert best_path.exists()
    assert trials_path.exists()
    assert rerank_path.exists()

    best = json.loads(best_path.read_text(encoding="utf-8"))
    assert "overrides" in best
    assert "scenarios" in best
    assert "aegis_realism_score" in best
    assert len(best["scenarios"]) == 1
