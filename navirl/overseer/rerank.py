from __future__ import annotations

import json

from navirl.overseer.provider import (
    ProviderCallError,
    ProviderConfig,
    ProviderUnavailableError,
    run_structured_vlm,
)

RERANK_SCHEMA = {
    "type": "object",
    "required": ["ranking"],
    "properties": {
        "ranking": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "notes": {"type": "string"},
    },
}


def _scenario_realism_score(s: dict) -> float:
    metrics = dict(s.get("metrics", {}))
    judge_conf = float(s.get("judge_confidence", 0.0))
    inv_pass = bool(s.get("invariants_pass", False))
    judge_pass = bool(s.get("judge_pass", False))

    horizon = max(1, int(metrics.get("horizon_steps", 1)))
    pair_rate = float(metrics.get("collisions_agent_agent", 0.0)) / float(horizon)
    obstacle_collisions = float(metrics.get("collisions_agent_obstacle", 0.0))
    deadlock = float(metrics.get("deadlock_count", 0.0))
    intrusion = float(metrics.get("intrusion_rate", 0.0))
    jerk = float(metrics.get("jerk_proxy", 0.0))
    oscillation = float(metrics.get("oscillation_score", 0.0))
    success = float(metrics.get("success_rate", 0.0))
    retry_count = float(metrics.get("_retry_count", 0.0))

    score = 0.0
    score += 1.8 * judge_conf
    score += 1.0 * success
    score -= 3.5 * deadlock
    score -= 1.8 * obstacle_collisions
    score -= 1.1 * intrusion
    score -= 0.9 * pair_rate
    score -= 0.25 * oscillation
    score -= 0.04 * jerk
    score -= 0.35 * retry_count
    if not inv_pass:
        score -= 2.0
    if not judge_pass:
        score -= 1.5
    return float(score)


def _trial_realism_score(trial: dict) -> float:
    scenarios = [s for s in trial.get("scenarios", []) if isinstance(s, dict)]
    if not scenarios:
        return -100.0
    total = sum(_scenario_realism_score(s) for s in scenarios)
    return float(total / len(scenarios))


def _rerank_prompt(candidates: list[dict]) -> str:
    payload = {
        "agent_name": "Aegis Overseer",
        "task": "rank_trials_by_visual_realism",
        "instructions": [
            "Return trial indices ordered best to worst for realism.",
            "Prioritize smooth trajectories, plausible speed, low deadlock/wall sticking, and social realism.",
            "Return strict JSON only.",
        ],
        "response_schema": RERANK_SCHEMA,
        "candidates": candidates,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _parse_vlm_ranking(payload: dict, candidate_indices: set[int]) -> list[int]:
    ranking = payload.get("ranking", [])
    if not isinstance(ranking, list):
        return []
    out = []
    for val in ranking:
        try:
            idx = int(val)
        except (TypeError, ValueError):
            continue
        if idx in candidate_indices and idx not in out:
            out.append(idx)
    return out


def run_aegis_rerank(
    ranking: list[dict],
    *,
    mode: str,
    provider_config: ProviderConfig,
    top_k: int = 6,
    allow_fallback: bool = True,
) -> dict:
    heuristic_scores = {int(t["trial_idx"]): _trial_realism_score(t) for t in ranking}
    blended_scores = dict(heuristic_scores)

    provider_used = False
    provider_error = ""
    provider_ranking: list[int] = []

    if str(mode).lower() == "vlm":
        candidate_trials = ranking[: max(1, int(top_k))]
        candidate_indices = {int(t["trial_idx"]) for t in candidate_trials}
        compact_candidates = []
        for t in candidate_trials:
            compact_candidates.append(
                {
                    "trial_idx": int(t["trial_idx"]),
                    "aggregate_score": float(t.get("aggregate_score", 0.0)),
                    "pass_rate": float(t.get("pass_rate", 0.0)),
                    "mean_judge_confidence": float(t.get("mean_judge_confidence", 0.0)),
                    "aegis_heuristic_realism": float(heuristic_scores[int(t["trial_idx"])]),
                    "scenarios": [
                        {
                            "scenario_id": str(s.get("scenario_id", "")),
                            "judge_confidence": float(s.get("judge_confidence", 0.0)),
                            "invariants_pass": bool(s.get("invariants_pass", False)),
                            "judge_pass": bool(s.get("judge_pass", False)),
                            "metrics": {
                                "deadlock_count": float(
                                    s.get("metrics", {}).get("deadlock_count", 0.0)
                                ),
                                "intrusion_rate": float(
                                    s.get("metrics", {}).get("intrusion_rate", 0.0)
                                ),
                                "collisions_agent_obstacle": float(
                                    s.get("metrics", {}).get("collisions_agent_obstacle", 0.0)
                                ),
                                "collisions_agent_agent": float(
                                    s.get("metrics", {}).get("collisions_agent_agent", 0.0)
                                ),
                                "oscillation_score": float(
                                    s.get("metrics", {}).get("oscillation_score", 0.0)
                                ),
                                "jerk_proxy": float(s.get("metrics", {}).get("jerk_proxy", 0.0)),
                                "success_rate": float(
                                    s.get("metrics", {}).get("success_rate", 0.0)
                                ),
                            },
                        }
                        for s in t.get("scenarios", [])
                        if isinstance(s, dict)
                    ],
                }
            )

        try:
            payload = run_structured_vlm(
                prompt=_rerank_prompt(compact_candidates),
                image_paths=[],
                schema=RERANK_SCHEMA,
                config=provider_config,
            )
            provider_ranking = _parse_vlm_ranking(payload, candidate_indices)
            provider_used = bool(provider_ranking)
        except (ProviderUnavailableError, ProviderCallError, ValueError) as exc:
            provider_error = str(exc)
            if not allow_fallback:
                return {
                    "applied": False,
                    "provider_used": False,
                    "provider_error": provider_error,
                    "heuristic_scores": heuristic_scores,
                    "blended_scores": blended_scores,
                    "status": "needs_human_review",
                }

    if provider_ranking:
        n = len(provider_ranking)
        for rank, trial_idx in enumerate(provider_ranking):
            if n <= 1:
                boost = 1.0
            else:
                boost = 1.4 * (1.0 - (rank / (n - 1)))
            blended_scores[trial_idx] = float(blended_scores.get(trial_idx, 0.0) + boost)

    return {
        "applied": True,
        "provider_used": provider_used,
        "provider_error": provider_error,
        "provider_ranking": provider_ranking,
        "heuristic_scores": heuristic_scores,
        "blended_scores": blended_scores,
        "status": "ok",
    }
