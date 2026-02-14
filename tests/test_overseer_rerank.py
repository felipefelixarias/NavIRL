from __future__ import annotations

from navirl.overseer import ProviderConfig, run_aegis_rerank


def _ranking_fixture() -> list[dict]:
    return [
        {
            "trial_idx": 0,
            "aggregate_score": 2.0,
            "pass_rate": 1.0,
            "mean_judge_confidence": 0.9,
            "scenarios": [
                {
                    "scenario_id": "s0",
                    "judge_confidence": 0.9,
                    "invariants_pass": True,
                    "judge_pass": True,
                    "metrics": {
                        "horizon_steps": 40,
                        "collisions_agent_agent": 0,
                        "collisions_agent_obstacle": 0,
                        "deadlock_count": 0,
                        "intrusion_rate": 0.1,
                        "jerk_proxy": 0.15,
                        "oscillation_score": 0.02,
                        "success_rate": 1.0,
                        "_retry_count": 0,
                    },
                }
            ],
        },
        {
            "trial_idx": 1,
            "aggregate_score": -1.0,
            "pass_rate": 0.0,
            "mean_judge_confidence": 0.2,
            "scenarios": [
                {
                    "scenario_id": "s0",
                    "judge_confidence": 0.2,
                    "invariants_pass": False,
                    "judge_pass": False,
                    "metrics": {
                        "horizon_steps": 40,
                        "collisions_agent_agent": 14,
                        "collisions_agent_obstacle": 3,
                        "deadlock_count": 1,
                        "intrusion_rate": 0.95,
                        "jerk_proxy": 0.95,
                        "oscillation_score": 0.7,
                        "success_rate": 0.0,
                        "_retry_count": 3,
                    },
                }
            ],
        },
    ]


def test_aegis_rerank_heuristic_mode_scores_trials():
    ranking = _ranking_fixture()
    out = run_aegis_rerank(
        ranking,
        mode="heuristic",
        provider_config=ProviderConfig(provider="codex"),
        top_k=2,
    )

    assert out["status"] == "ok"
    assert out["applied"] is True
    assert out["provider_used"] is False
    assert out["blended_scores"][0] > out["blended_scores"][1]


def test_aegis_rerank_vlm_without_fallback_requests_human_review():
    ranking = _ranking_fixture()
    out = run_aegis_rerank(
        ranking,
        mode="vlm",
        provider_config=ProviderConfig(provider="codex"),
        top_k=2,
        allow_fallback=False,
    )

    assert out["status"] == "needs_human_review"
    assert out["provider_used"] is False
