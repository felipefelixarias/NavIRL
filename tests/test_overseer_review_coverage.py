"""Coverage-focused tests for navirl.overseer.review internals.

These tests exercise private helpers and branches that the existing
``test_overseer_review.py`` does not reach:

* ``_load_bundle_scenario`` missing-file, malformed-YAML and non-dict paths
* ``_state_speed_ratio_stats`` with populated agents and with empty state
* ``_confidence_penalty`` for each severity class
* ``_normalize_provider_review`` status and confidence clamping
* ``_dedupe_violations`` duplicate suppression
* ``build_aegis_findings`` branches for speed spikes, stalls, wall hugging,
  jitter-not-severe-enough, robot/human scale mismatch
* ``run_aegis_review`` happy-path with a mocked provider (exercises the
  provider_used branch, combined-confidence logic and recommendations).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from navirl.overseer import ProviderConfig, run_aegis_review
from navirl.overseer.review import (
    AEGIS_REVIEW_SCHEMA,
    _confidence_penalty,
    _dedupe_violations,
    _load_bundle_scenario,
    _normalize_provider_review,
    _state_speed_ratio_stats,
    build_aegis_findings,
)

# ---------------------------------------------------------------------------
#  _load_bundle_scenario
# ---------------------------------------------------------------------------


class TestLoadBundleScenario:
    def test_missing_file_returns_empty_dict(self, tmp_path: Path):
        assert _load_bundle_scenario(tmp_path) == {}

    def test_malformed_yaml_returns_empty_dict(self, tmp_path: Path):
        (tmp_path / "scenario.yaml").write_text(
            "robot: {radius: [unterminated", encoding="utf-8"
        )
        # Invalid YAML should be swallowed into an empty dict.
        assert _load_bundle_scenario(tmp_path) == {}

    def test_non_dict_yaml_returns_empty_dict(self, tmp_path: Path):
        # A list at the top level is valid YAML but not a mapping.
        (tmp_path / "scenario.yaml").write_text(
            yaml.safe_dump(["a", "b", "c"]), encoding="utf-8"
        )
        assert _load_bundle_scenario(tmp_path) == {}

    def test_non_utf8_bytes_returns_empty_dict(self, tmp_path: Path):
        # Write a byte that is not valid UTF-8.
        (tmp_path / "scenario.yaml").write_bytes(b"\xff\xfe not valid utf-8")
        assert _load_bundle_scenario(tmp_path) == {}

    def test_valid_dict_passes_through(self, tmp_path: Path):
        payload = {"robot": {"radius": 0.2}, "humans": {"count": 3}}
        (tmp_path / "scenario.yaml").write_text(
            yaml.safe_dump(payload), encoding="utf-8"
        )
        out = _load_bundle_scenario(tmp_path)
        assert out == payload


# ---------------------------------------------------------------------------
#  _state_speed_ratio_stats
# ---------------------------------------------------------------------------


class TestStateSpeedRatioStats:
    def test_missing_file_returns_zero_stats(self, tmp_path: Path):
        stats = _state_speed_ratio_stats(tmp_path)
        assert stats == {"max_ratio": 0.0, "mean_ratio": 0.0, "num_samples": 0}

    def test_empty_agents_returns_zero_stats(self, tmp_path: Path):
        (tmp_path / "state.jsonl").write_text(
            json.dumps({"agents": []}) + "\n", encoding="utf-8"
        )
        stats = _state_speed_ratio_stats(tmp_path)
        assert stats == {"max_ratio": 0.0, "mean_ratio": 0.0, "num_samples": 0}

    def test_computes_max_and_mean(self, tmp_path: Path):
        rows = [
            {
                "agents": [
                    {"vx": 0.4, "vy": 0.3, "max_speed": 1.0},  # speed 0.5, ratio 0.5
                    {"vx": 0.0, "vy": 0.0, "max_speed": 1.0},  # ratio 0
                ]
            },
            {
                "agents": [
                    {"vx": 0.6, "vy": 0.8, "max_speed": 1.0},  # speed 1.0, ratio 1.0
                ]
            },
        ]
        path = tmp_path / "state.jsonl"
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

        stats = _state_speed_ratio_stats(tmp_path)
        assert stats["num_samples"] == 3
        assert stats["max_ratio"] == 1.0
        # mean of (0.5, 0.0, 1.0) = 0.5
        assert abs(stats["mean_ratio"] - 0.5) < 1e-9

    def test_missing_max_speed_uses_1_0_default(self, tmp_path: Path):
        # If max_speed absent, defaults to 1.0 via get(); zero-valued max_speed
        # is guarded with a 1e-6 floor.
        row = {"agents": [{"vx": 3.0, "vy": 4.0, "max_speed": 0.0}]}
        path = tmp_path / "state.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        stats = _state_speed_ratio_stats(tmp_path)
        # speed = 5.0, max_speed floored to 1e-6 → ratio is enormous
        assert stats["max_ratio"] > 1e5

    def test_skips_blank_lines(self, tmp_path: Path):
        text = (
            json.dumps({"agents": [{"vx": 0.3, "vy": 0.4, "max_speed": 1.0}]})
            + "\n\n\n"
        )
        (tmp_path / "state.jsonl").write_text(text, encoding="utf-8")
        stats = _state_speed_ratio_stats(tmp_path)
        assert stats["num_samples"] == 1
        assert abs(stats["max_ratio"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
#  _confidence_penalty
# ---------------------------------------------------------------------------


class TestConfidencePenalty:
    def test_blocker_penalty(self):
        assert abs(_confidence_penalty([{"severity": "blocker"}]) - 0.18) < 1e-9

    def test_major_penalty(self):
        assert abs(_confidence_penalty([{"severity": "major"}]) - 0.08) < 1e-9

    def test_minor_or_unknown_penalty(self):
        assert abs(_confidence_penalty([{"severity": "minor"}]) - 0.03) < 1e-9
        # Anything not "blocker"/"major" falls into the else branch.
        assert abs(_confidence_penalty([{"severity": "info"}]) - 0.03) < 1e-9
        assert abs(_confidence_penalty([{}]) - 0.03) < 1e-9  # defaults to "minor"

    def test_additive_mix(self):
        vs = [{"severity": "blocker"}, {"severity": "major"}, {"severity": "minor"}]
        # 0.18 + 0.08 + 0.03 = 0.29
        assert abs(_confidence_penalty(vs) - 0.29) < 1e-9

    def test_empty_list_zero(self):
        assert _confidence_penalty([]) == 0.0


# ---------------------------------------------------------------------------
#  _normalize_provider_review
# ---------------------------------------------------------------------------


class TestNormalizeProviderReview:
    def test_defaults_on_empty_payload(self):
        out = _normalize_provider_review({})
        assert out["overall_pass"] is False
        assert out["confidence"] == 0.0
        assert out["status"] == "fail"
        assert out["violations"] == []

    def test_clamps_confidence_above_one(self):
        out = _normalize_provider_review({"confidence": 2.5, "status": "pass"})
        assert out["confidence"] == 1.0
        assert out["status"] == "pass"

    def test_clamps_confidence_below_zero(self):
        out = _normalize_provider_review({"confidence": -0.5})
        assert out["confidence"] == 0.0

    def test_unknown_status_becomes_fail(self):
        out = _normalize_provider_review({"status": "nonsense_status"})
        assert out["status"] == "fail"

    def test_preserves_valid_status(self):
        for status in ("pass", "fail", "needs_human_review"):
            out = _normalize_provider_review({"status": status})
            assert out["status"] == status

    def test_non_list_violations_reset(self):
        out = _normalize_provider_review({"violations": "not a list"})
        assert out["violations"] == []

    def test_list_violations_preserved(self):
        viol = [{"type": "x", "evidence": "y", "severity": "minor"}]
        out = _normalize_provider_review({"violations": viol})
        assert out["violations"] == viol

    def test_overall_pass_coerced_to_bool(self):
        out = _normalize_provider_review({"overall_pass": 1})
        assert out["overall_pass"] is True
        out2 = _normalize_provider_review({"overall_pass": 0})
        assert out2["overall_pass"] is False


# ---------------------------------------------------------------------------
#  _dedupe_violations
# ---------------------------------------------------------------------------


class TestDedupeViolations:
    def test_removes_exact_duplicates(self):
        vs = [
            {"type": "a", "evidence": "e1", "severity": "blocker"},
            {"type": "a", "evidence": "e1", "severity": "major"},  # dup by (type,evidence)
            {"type": "b", "evidence": "e1", "severity": "minor"},
        ]
        out = _dedupe_violations(vs)
        assert len(out) == 2
        assert {v["type"] for v in out} == {"a", "b"}

    def test_preserves_order_of_first_seen(self):
        vs = [
            {"type": "first", "evidence": "x"},
            {"type": "second", "evidence": "y"},
            {"type": "first", "evidence": "x"},
        ]
        out = _dedupe_violations(vs)
        assert [v["type"] for v in out] == ["first", "second"]

    def test_empty_input(self):
        assert _dedupe_violations([]) == []


# ---------------------------------------------------------------------------
#  build_aegis_findings — extra branches
# ---------------------------------------------------------------------------


def _write_state_with_speed_ratio(bundle_dir: Path, ratio: float) -> None:
    """Write a one-row state.jsonl whose single agent has the given speed/max_speed ratio."""
    (bundle_dir / "state.jsonl").write_text(
        json.dumps({"agents": [{"vx": ratio, "vy": 0.0, "max_speed": 1.0}]}) + "\n",
        encoding="utf-8",
    )


def _minimal_summary_no_issues() -> dict:
    return {
        "scenario_id": "clean",
        "metrics": {},
        "map": {"width_m": 5.0, "height_m": 5.0},
        "invariants": {"checks": []},
    }


class TestBuildAegisFindingsBranches:
    def test_extreme_speed_spike_blocker(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_extreme"
        bundle_dir.mkdir()
        _write_state_with_speed_ratio(bundle_dir, 1.4)

        findings = build_aegis_findings(bundle_dir, _minimal_summary_no_issues())
        types = [f["type"] for f in findings]
        assert "extreme_speed_spike" in types
        f = next(f for f in findings if f["type"] == "extreme_speed_spike")
        assert f["severity"] == "blocker"

    def test_speed_near_unrealistic_major(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_near"
        bundle_dir.mkdir()
        _write_state_with_speed_ratio(bundle_dir, 1.15)

        findings = build_aegis_findings(bundle_dir, _minimal_summary_no_issues())
        types = [f["type"] for f in findings]
        assert "speed_near_unrealistic" in types
        assert "extreme_speed_spike" not in types

    def test_safe_speed_no_spike_finding(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_safe"
        bundle_dir.mkdir()
        _write_state_with_speed_ratio(bundle_dir, 1.0)

        findings = build_aegis_findings(bundle_dir, _minimal_summary_no_issues())
        types = [f["type"] for f in findings]
        assert "speed_near_unrealistic" not in types
        assert "extreme_speed_spike" not in types

    def test_motion_jitter_below_threshold_produces_no_finding(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_calm"
        bundle_dir.mkdir()

        summary = _minimal_summary_no_issues()
        # worst=0.5, max=1.0 → 0.5 <= 0.9 → no finding
        summary["invariants"]["checks"] = [
            {"name": "motion_jitter", "worst_flip_rate": 0.5, "max_flip_rate": 1.0}
        ]

        findings = build_aegis_findings(bundle_dir, summary)
        assert not any(f["type"] == "unnatural_jitter" for f in findings)

    def test_motion_jitter_major_severity(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_jittery_major"
        bundle_dir.mkdir()

        summary = _minimal_summary_no_issues()
        # worst=0.92, max=1.0 → above 0.9*max but at/below max → major
        summary["invariants"]["checks"] = [
            {"name": "motion_jitter", "worst_flip_rate": 0.92, "max_flip_rate": 1.0}
        ]

        findings = build_aegis_findings(bundle_dir, summary)
        major = next(f for f in findings if f["type"] == "unnatural_jitter")
        assert major["severity"] == "major"

    def test_near_wall_or_goal_stall_major(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_stall"
        bundle_dir.mkdir()

        summary = _minimal_summary_no_issues()
        summary["invariants"]["checks"] = [
            {
                "name": "agent_stop_duration",
                "max_stop_seconds": 2.0,
                "top_longest_stops": [{"agent_id": "h0", "max_stopped_seconds": 1.8}],
            }
        ]

        findings = build_aegis_findings(bundle_dir, summary)
        assert any(f["type"] == "near_wall_or_goal_stall" for f in findings)

    def test_excess_wall_hugging_major(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_wall"
        bundle_dir.mkdir()

        summary = _minimal_summary_no_issues()
        summary["invariants"]["checks"] = [
            {
                "name": "wall_proximity_fraction",
                "near_wall_fraction": 0.25,  # above 0.9*0.25 limit
                "max_fraction": 0.25,
            }
        ]

        findings = build_aegis_findings(bundle_dir, summary)
        assert any(f["type"] == "excess_wall_hugging" for f in findings)

    def test_robot_human_scale_mismatch_upper(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_ratio_big"
        bundle_dir.mkdir()
        # Use a small-but-plausible robot radius vs a very small human so the
        # ratio blows past 3.5 without triggering robot_scale_implausible.
        (bundle_dir / "scenario.yaml").write_text(
            yaml.safe_dump({"robot": {"radius": 0.3}, "humans": {"radius": 0.05}}),
            encoding="utf-8",
        )

        findings = build_aegis_findings(bundle_dir, _minimal_summary_no_issues())
        types = [f["type"] for f in findings]
        assert "robot_human_scale_mismatch" in types
        assert "robot_scale_implausible" not in types

    def test_robot_human_scale_mismatch_lower(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_ratio_small"
        bundle_dir.mkdir()
        (bundle_dir / "scenario.yaml").write_text(
            yaml.safe_dump({"robot": {"radius": 0.1}, "humans": {"radius": 0.3}}),
            encoding="utf-8",
        )

        findings = build_aegis_findings(bundle_dir, _minimal_summary_no_issues())
        assert any(f["type"] == "robot_human_scale_mismatch" for f in findings)

    def test_robot_scale_implausible(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_big_robot"
        bundle_dir.mkdir()

        summary = _minimal_summary_no_issues()
        summary["map"] = {"width_m": 3.0, "height_m": 3.0}
        # 0.22 * 3.0 = 0.66, robot radius above that triggers blocker.
        (bundle_dir / "scenario.yaml").write_text(
            yaml.safe_dump({"robot": {"radius": 0.8}, "humans": {"radius": 0.5}}),
            encoding="utf-8",
        )

        findings = build_aegis_findings(bundle_dir, summary)
        assert any(f["type"] == "robot_scale_implausible" for f in findings)

    def test_unsafe_robot_human_clearance(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_unsafe"
        bundle_dir.mkdir()

        summary = _minimal_summary_no_issues()
        summary["metrics"] = {"min_dist_robot_human_min": 0.05}

        findings = build_aegis_findings(bundle_dir, summary)
        assert any(f["type"] == "unsafe_robot_human_clearance" for f in findings)

    def test_no_findings_on_clean_bundle(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_clean"
        bundle_dir.mkdir()
        (bundle_dir / "scenario.yaml").write_text(
            yaml.safe_dump({"robot": {"radius": 0.2}, "humans": {"radius": 0.18}}),
            encoding="utf-8",
        )

        summary = _minimal_summary_no_issues()
        summary["map"] = {"width_m": 10.0, "height_m": 10.0}
        summary["metrics"] = {"min_dist_robot_human_min": 0.5}

        findings = build_aegis_findings(bundle_dir, summary)
        assert findings == []


# ---------------------------------------------------------------------------
#  run_aegis_review — provider-used branch
# ---------------------------------------------------------------------------


class _StubProvider:
    """Callable that records prompts and returns a fixed normalized payload."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.prompts: list[str] = []
        self.images: list[list[str]] = []

    def __call__(self, *, prompt, image_paths, schema, config) -> dict:
        assert schema == AEGIS_REVIEW_SCHEMA
        assert isinstance(config, ProviderConfig)
        self.prompts.append(prompt)
        self.images.append(list(image_paths))
        return self.payload


def _summary_for_provider_test() -> dict:
    return {
        "scenario_id": "provider_test",
        "metrics": {
            "deadlock_count": 0,
            "collisions_agent_obstacle": 0,
            "min_dist_robot_human_min": 0.4,
        },
        "map": {"width_m": 6.0, "height_m": 6.0, "pixels_per_meter": 100.0},
        "invariants": {"checks": []},
    }


class TestRunAegisReviewProviderUsed:
    def test_provider_payload_blends_confidence(self, tmp_path: Path, monkeypatch):
        bundle_dir = tmp_path / "bundle_ok"
        bundle_dir.mkdir()
        (bundle_dir / "scenario.yaml").write_text(
            yaml.safe_dump({"robot": {"radius": 0.2}, "humans": {"radius": 0.18}}),
            encoding="utf-8",
        )
        (bundle_dir / "state.jsonl").write_text(
            json.dumps({"agents": []}) + "\n", encoding="utf-8"
        )

        stub = _StubProvider(
            {
                "overall_pass": True,
                "confidence": 0.9,
                "status": "pass",
                "violations": [],
            }
        )
        monkeypatch.setattr("navirl.overseer.review.run_structured_vlm", stub)

        heuristic = {
            "overall_pass": True,
            "confidence": 0.7,
            "violations": [],
            "status": "pass",
        }

        payload = run_aegis_review(
            bundle_dir=bundle_dir,
            summary=_summary_for_provider_test(),
            frame_paths=["frame1.png", "frame2.png"],
            heuristic_payload=heuristic,
            provider_config=ProviderConfig(provider="codex"),
            confidence_threshold=0.6,
            require_video=True,
            allow_fallback=True,
        )

        assert payload["judge_type"] == "aegis_vlm_hybrid"
        assert payload["provider_trace"]["provider_used"] is True
        assert payload["provider_trace"]["fallback_used"] is False
        # Confidence blends 0.55 * provider + 0.45 * heuristic. No findings
        # produced -> no finding-penalty applied. Expected ≈ 0.55*0.9 + 0.45*0.7 = 0.81
        assert abs(payload["confidence"] - (0.55 * 0.9 + 0.45 * 0.7)) < 1e-9
        assert payload["overall_pass"] is True
        assert payload["status"] == "pass"
        # Exactly one prompt was dispatched and images were forwarded.
        assert len(stub.prompts) == 1
        # Payload is valid JSON containing our frame list.
        parsed = json.loads(stub.prompts[0])
        assert parsed["frame_paths"] == ["frame1.png", "frame2.png"]
        assert parsed["agent_name"] == "Aegis Overseer"
        assert parsed["require_video"] is True

    def test_provider_blocker_forces_fail(self, tmp_path: Path, monkeypatch):
        bundle_dir = tmp_path / "bundle_provider_fail"
        bundle_dir.mkdir()
        (bundle_dir / "scenario.yaml").write_text(
            yaml.safe_dump({"robot": {"radius": 0.2}, "humans": {"radius": 0.18}}),
            encoding="utf-8",
        )

        stub = _StubProvider(
            {
                "overall_pass": False,
                "confidence": 0.95,
                "status": "fail",
                "violations": [
                    {"type": "vlm_wall_stick", "evidence": "clips", "severity": "blocker"}
                ],
            }
        )
        monkeypatch.setattr("navirl.overseer.review.run_structured_vlm", stub)

        payload = run_aegis_review(
            bundle_dir=bundle_dir,
            summary=_summary_for_provider_test(),
            frame_paths=[],
            heuristic_payload={
                "overall_pass": True,
                "confidence": 0.95,
                "violations": [],
            },
            provider_config=ProviderConfig(provider="codex"),
            confidence_threshold=0.5,
            require_video=False,
            allow_fallback=False,
        )

        # Any blocker-severity violation from the provider forces overall fail
        # regardless of confidence.
        assert payload["overall_pass"] is False
        assert payload["status"] == "fail"
        assert payload["judge_type"] == "aegis_vlm_hybrid"
        assert any(
            v.get("type") == "vlm_wall_stick" for v in payload["violations"]
        )

    def test_provider_dedupes_against_heuristic_violations(
        self, tmp_path: Path, monkeypatch
    ):
        bundle_dir = tmp_path / "bundle_dup"
        bundle_dir.mkdir()
        (bundle_dir / "scenario.yaml").write_text(
            yaml.safe_dump({"robot": {"radius": 0.2}, "humans": {"radius": 0.18}}),
            encoding="utf-8",
        )

        shared_violation = {
            "type": "shared_type",
            "evidence": "same evidence",
            "severity": "minor",
        }
        stub = _StubProvider(
            {
                "overall_pass": True,
                "confidence": 0.8,
                "status": "pass",
                "violations": [shared_violation],
            }
        )
        monkeypatch.setattr("navirl.overseer.review.run_structured_vlm", stub)

        payload = run_aegis_review(
            bundle_dir=bundle_dir,
            summary=_summary_for_provider_test(),
            frame_paths=[],
            heuristic_payload={
                "overall_pass": True,
                "confidence": 0.8,
                "violations": [shared_violation],
            },
            provider_config=ProviderConfig(provider="codex"),
            confidence_threshold=0.5,
            require_video=False,
            allow_fallback=True,
        )

        # The duplicate appears only once in the merged result.
        matching = [v for v in payload["violations"] if v["type"] == "shared_type"]
        assert len(matching) == 1

    def test_provider_exception_falls_back(self, tmp_path: Path, monkeypatch):
        bundle_dir = tmp_path / "bundle_fallback"
        bundle_dir.mkdir()
        (bundle_dir / "scenario.yaml").write_text(
            yaml.safe_dump({"robot": {"radius": 0.2}, "humans": {"radius": 0.18}}),
            encoding="utf-8",
        )

        from navirl.overseer.provider import ProviderCallError

        def boom(**_kw: Any) -> dict:
            raise ProviderCallError("simulated transient failure")

        monkeypatch.setattr("navirl.overseer.review.run_structured_vlm", boom)

        payload = run_aegis_review(
            bundle_dir=bundle_dir,
            summary=_summary_for_provider_test(),
            frame_paths=[],
            heuristic_payload={
                "overall_pass": True,
                "confidence": 0.6,
                "violations": [],
            },
            provider_config=ProviderConfig(provider="codex"),
            confidence_threshold=0.5,
            require_video=False,
            allow_fallback=True,
        )

        assert payload["judge_type"] == "aegis_vlm_fallback_heuristic"
        assert payload["provider_trace"]["fallback_used"] is True
        assert payload["provider_trace"]["provider_used"] is False
        assert "simulated transient failure" in payload["provider_trace"]["error"]
