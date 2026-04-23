"""Tests for navirl.overseer.rerank uncovered paths.

The baseline ``test_overseer_rerank`` only covers the heuristic-only
path and the fallback-disabled VLM error path. This file extends
coverage to:

- ``_trial_realism_score`` degenerate case (empty scenarios)
- ``_parse_vlm_ranking`` input validation
- ``run_aegis_rerank`` VLM success path that blends the provider ranking
  into ``blended_scores``
- ``run_aegis_rerank`` VLM success path with a single candidate
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from navirl.overseer.rerank import (
    _parse_vlm_ranking,
    _trial_realism_score,
    run_aegis_rerank,
)

try:
    from navirl.overseer import ProviderConfig
except ImportError:
    from navirl.overseer.provider import ProviderConfig


def _trial(trial_idx: int, *, pass_rate: float = 1.0, judge_conf: float = 0.9) -> dict:
    return {
        "trial_idx": trial_idx,
        "aggregate_score": 2.0 if pass_rate > 0.5 else -1.0,
        "pass_rate": pass_rate,
        "mean_judge_confidence": judge_conf,
        "scenarios": [
            {
                "scenario_id": f"s{trial_idx}",
                "judge_confidence": judge_conf,
                "invariants_pass": pass_rate > 0.5,
                "judge_pass": pass_rate > 0.5,
                "metrics": {
                    "horizon_steps": 40,
                    "collisions_agent_agent": 0,
                    "collisions_agent_obstacle": 0,
                    "deadlock_count": 0,
                    "intrusion_rate": 0.0,
                    "jerk_proxy": 0.1,
                    "oscillation_score": 0.0,
                    "success_rate": pass_rate,
                    "_retry_count": 0,
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# _trial_realism_score
# ---------------------------------------------------------------------------


class TestTrialRealismScore:
    def test_empty_scenarios_returns_floor(self):
        """A trial with no scenarios gets the -100.0 floor score."""
        assert _trial_realism_score({"scenarios": []}) == -100.0

    def test_missing_scenarios_key_returns_floor(self):
        assert _trial_realism_score({}) == -100.0

    def test_filters_non_dict_scenarios(self):
        """Non-dict entries in scenarios are filtered out before averaging."""
        trial = {
            "scenarios": [
                None,
                "not_a_dict",
                42,
            ]
        }
        assert _trial_realism_score(trial) == -100.0

    def test_averages_scenario_scores(self):
        # Two scenarios with identical metrics should produce a stable
        # non-floor score.
        trial = _trial(0, pass_rate=1.0, judge_conf=0.9)
        score = _trial_realism_score(trial)
        assert score > -100.0
        assert score > 0.0  # high judge confidence + success


# ---------------------------------------------------------------------------
# _parse_vlm_ranking
# ---------------------------------------------------------------------------


class TestParseVlmRanking:
    def test_valid_ranking_subset(self):
        payload = {"ranking": [2, 0, 1]}
        out = _parse_vlm_ranking(payload, candidate_indices={0, 1, 2})
        assert out == [2, 0, 1]

    def test_ranking_not_a_list_returns_empty(self):
        assert _parse_vlm_ranking({"ranking": "not_a_list"}, {0, 1}) == []

    def test_missing_ranking_key_returns_empty(self):
        assert _parse_vlm_ranking({}, {0, 1}) == []

    def test_drops_unknown_indices(self):
        """Indices not in candidate_indices are skipped."""
        payload = {"ranking": [0, 99, 1, 42]}
        out = _parse_vlm_ranking(payload, candidate_indices={0, 1})
        assert out == [0, 1]

    def test_deduplicates_repeated_indices(self):
        payload = {"ranking": [0, 1, 0, 1, 0]}
        out = _parse_vlm_ranking(payload, candidate_indices={0, 1})
        assert out == [0, 1]

    def test_non_integer_values_are_skipped(self):
        """TypeError / ValueError on int() conversion is swallowed."""
        payload = {"ranking": [0, "bad", None, 1, 2.5]}
        out = _parse_vlm_ranking(payload, candidate_indices={0, 1, 2})
        # 2.5 casts to 2; "bad" and None are dropped.
        assert out == [0, 1, 2]


# ---------------------------------------------------------------------------
# run_aegis_rerank — VLM success path
# ---------------------------------------------------------------------------


class TestRerunAegisRerankVlm:
    def test_vlm_success_applies_provider_boost(self):
        """When the VLM returns a valid ranking, blended_scores gain a
        positional boost (lines 165-166 and 180-186)."""
        ranking = [
            _trial(0, pass_rate=0.0, judge_conf=0.1),  # worst heuristically
            _trial(1, pass_rate=1.0, judge_conf=0.9),  # best heuristically
            _trial(2, pass_rate=0.5, judge_conf=0.5),
        ]

        with patch("navirl.overseer.rerank.run_structured_vlm") as mock_vlm:
            # VLM disagrees with the heuristic and ranks trial 0 best.
            mock_vlm.return_value = {"ranking": [0, 2, 1]}
            result = run_aegis_rerank(
                ranking,
                mode="vlm",
                provider_config=ProviderConfig(provider="codex"),
                top_k=3,
            )

        assert result["status"] == "ok"
        assert result["provider_used"] is True
        assert result["provider_error"] == ""
        assert result["provider_ranking"] == [0, 2, 1]
        # Trial 0 should have received the highest rank boost (+1.4),
        # lifting it above its raw heuristic score.
        heur = result["heuristic_scores"]
        blended = result["blended_scores"]
        assert blended[0] > heur[0]
        # The last-ranked trial (1) only gets a 0.0 boost.
        assert blended[1] == pytest.approx(heur[1], abs=1e-9)

    def test_vlm_success_with_single_candidate_gives_full_boost(self):
        """A single-candidate ranking takes the n<=1 branch (boost=1.0)."""
        ranking = [_trial(0, pass_rate=1.0, judge_conf=0.9)]

        with patch("navirl.overseer.rerank.run_structured_vlm") as mock_vlm:
            mock_vlm.return_value = {"ranking": [0]}
            result = run_aegis_rerank(
                ranking,
                mode="vlm",
                provider_config=ProviderConfig(provider="codex"),
                top_k=1,
            )

        assert result["provider_used"] is True
        assert result["provider_ranking"] == [0]
        assert result["blended_scores"][0] == pytest.approx(result["heuristic_scores"][0] + 1.0)

    def test_vlm_returns_empty_ranking_keeps_heuristic(self):
        """If _parse_vlm_ranking yields [], provider_used stays False
        and the heuristic scores pass through unchanged."""
        ranking = [_trial(0), _trial(1, pass_rate=0.0, judge_conf=0.1)]

        with patch("navirl.overseer.rerank.run_structured_vlm") as mock_vlm:
            # Ranking list of entirely-unknown indices → parsed as [].
            mock_vlm.return_value = {"ranking": [42, 99]}
            result = run_aegis_rerank(
                ranking,
                mode="vlm",
                provider_config=ProviderConfig(provider="codex"),
                top_k=2,
                allow_fallback=True,
            )

        assert result["provider_used"] is False
        assert result["provider_error"] == ""
        # Blended scores equal the heuristic scores when no boost applied.
        for idx, h in result["heuristic_scores"].items():
            assert result["blended_scores"][idx] == pytest.approx(h, abs=1e-9)

    def test_vlm_fallback_enabled_on_provider_error(self):
        """allow_fallback=True lets the pipeline continue on provider error,
        preserving heuristic scores."""
        from navirl.overseer.provider import ProviderCallError

        ranking = [_trial(0), _trial(1, pass_rate=0.0)]

        with patch(
            "navirl.overseer.rerank.run_structured_vlm",
            side_effect=ProviderCallError("boom"),
        ):
            result = run_aegis_rerank(
                ranking,
                mode="vlm",
                provider_config=ProviderConfig(provider="codex"),
                top_k=2,
                allow_fallback=True,
            )

        # Fallback: pipeline continues, status ok, error captured.
        assert result["status"] == "ok"
        assert result["provider_used"] is False
        assert "boom" in result["provider_error"]


# ---------------------------------------------------------------------------
# run_aegis_rerank — input-robustness edges
# ---------------------------------------------------------------------------


class TestRerunAegisRerankEdges:
    def test_top_k_zero_still_has_at_least_one_candidate(self):
        """``top_k`` is clamped to at least 1 inside the VLM branch."""
        ranking = [_trial(0)]
        with patch("navirl.overseer.rerank.run_structured_vlm") as mock_vlm:
            mock_vlm.return_value = {"ranking": [0]}
            result = run_aegis_rerank(
                ranking,
                mode="vlm",
                provider_config=ProviderConfig(provider="codex"),
                top_k=0,  # intentionally 0
            )
        assert result["status"] == "ok"
        assert result["provider_ranking"] == [0]

    def test_heuristic_mode_returns_unmodified_blended_scores(self):
        ranking = [_trial(0), _trial(1, pass_rate=0.0)]
        result = run_aegis_rerank(
            ranking,
            mode="heuristic",
            provider_config=ProviderConfig(provider="codex"),
            top_k=2,
        )
        # In heuristic mode no VLM interaction happens.
        assert result["provider_used"] is False
        assert result["provider_error"] == ""
        assert result["blended_scores"] == result["heuristic_scores"]
